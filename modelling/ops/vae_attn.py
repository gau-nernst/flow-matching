# inspired by FlashMLA
# https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md

from functools import cache

import cutlass
import torch
import torch.nn.functional as F
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int64, cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from gn_kernels.cutedsl.utils import EVICT_FIRST, mma_sync, permute, simple_tma_g2s

#                                |---DIM/2--|---DIM/2--|
#                               BK/2  V0L   |    V0R   |
#                                |----------|----------|
#                               BK/2  V1L   |    V1R   |
#                                |----------|----------|
# QK MMA
#      |    K0    |    K1    |   PV MMA
# -----|---BK/2---|---BK/2---|   |----------|----------|
# Q0 BQ/2         |          |   |          |          |
# -----|----------|----------|   |----------|----------|
# Q1 BQ/2         |          |   |          |          |
# -----|----------|----------|   |----------|----------|


class VaeAttn:
    # TODO: support BQ=64 / 4x2 warp
    DIM: int = 512
    BQ: int = 32
    BK: int = 128
    num_stages: int = 3

    @cute.jit
    def prepare_tma(
        self,
        x: cute.Tensor,
        BLOCK_L: cutlass.Constexpr[int],
        BLOCK_D: cutlass.Constexpr[int],
        num_stages: cutlass.Constexpr[int],
        tma_op: cpasync.TmaCopyOp,
    ):
        # QK: [B, L, D]
        swizzle = cute.make_swizzle(3, 4, 3)  # 128B
        s_layout = cute.make_layout(
            (1, BLOCK_L, (64, BLOCK_D // 64), num_stages),
            stride=(0, 64, (1, BLOCK_L * 64), BLOCK_L * BLOCK_D),
        )
        s_layout = cute.make_composed_layout(swizzle, 0, s_layout)
        return cpasync.make_tiled_tma_atom(tma_op, x, s_layout, (1, BLOCK_L, BLOCK_D))

    @cute.jit
    def __call__(self, gQ: cute.Tensor, gK: cute.Tensor, gV: cute.Tensor, gO: cute.Tensor, stream: CUstream):
        B, Lq, _ = gQ.shape

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        Q_tma = self.prepare_tma(gQ, self.BQ, self.DIM, 1, tma_g2s)
        K_tma = self.prepare_tma(gK, self.BK, 64, self.num_stages, tma_g2s)
        V_tma = self.prepare_tma(gV, 16, self.DIM, self.num_stages, tma_g2s)

        grid = (cute.ceil_div(Lq, self.BQ), B, 1)
        block = (5 * 32, 1, 1)
        self.kernel(Q_tma, K_tma, V_tma, gO).launch(grid=grid, block=block, stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, Q_tma: cpasync.TmaInfo, K_tma: cpasync.TmaInfo, V_tma: cpasync.TmaInfo, gO: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        q_tile_id, batch_id, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        Lk = K_tma.tma_tensor.shape[1]
        BQ = self.BQ
        BK = self.BK
        DIM = self.DIM
        WQ = BQ // 2
        WK = BK // 2
        num_stages = self.num_stages

        # allocate smem
        def allocate_smem(smem, s_layout):
            return smem.allocate_tensor(BFloat16, s_layout.outer, byte_alignment=128, swizzle=s_layout.inner)

        smem = cutlass.utils.SmemAllocator()
        sQ = allocate_smem(smem, Q_tma.smem_layout)[0, None, None, 0]  # 32 KiB
        sK = allocate_smem(smem, K_tma.smem_layout)[0, None, None, None]  # 32 KiB

        # V and K share the same slots
        assert cute.size(V_tma.smem_layout) == cute.size(sK)
        sV = cute.make_tensor(sK.iterator, V_tma.smem_layout.outer)[0, None, None, None]

        sP = smem.allocate_tensor(
            BFloat16,
            # cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, cute.make_layout((BK, BQ))),
            cute.make_layout((BK, BQ)),
            byte_alignment=128,
            swizzle=cute.make_swizzle(3, 4, 3),
        )  # 8 KiB

        # to compute rowmax
        sM = smem.allocate_tensor(Float32, cute.make_layout((2, WQ // 2, 4)))

        tma_q_mbar = smem.allocate_array(Int64, 1)
        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        BAR_MMA = 1

        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tma_q_mbar, 1)
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(Q_tma.atom)
            cpasync.prefetch_descriptor(K_tma.atom)
            cpasync.prefetch_descriptor(V_tma.atom)
        cute.arch.sync_threads()

        if warp_id == 4:
            # TMA warp
            # load Q
            gQ_tile = cute.local_tile(Q_tma.tma_tensor[batch_id, None, None], (BQ, DIM), (q_tile_id, 0))
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(tma_q_mbar, BQ * DIM * 2)
            simple_tma_g2s(Q_tma.atom, gQ_tile, sQ, tma_q_mbar, EVICT_FIRST)

            stage_id = 0
            parity = 1

            # K: [(BK, 64), (L/BK, HEAD_DIM/64)]
            # V: [(16, DIM), (L/16, 1)]
            gK_tiles = cute.zipped_divide(K_tma.tma_tensor[batch_id, None, None], (BK, 64))
            gV_tiles = cute.zipped_divide(V_tma.tma_tensor[batch_id, None, None], (16, DIM))

            for iter_l in range(cute.ceil_div(Lk, BK)):
                # load K for QK MMA
                for i in cutlass.range_constexpr(DIM // 64):
                    mbar = tma_full_mbar + stage_id
                    cute.arch.mbarrier_wait(tma_empty_mbar + stage_id, parity)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar, BK * 64 * 2)
                    simple_tma_g2s(K_tma.atom, gK_tiles[None, (iter_l, i)], sK[None, None, stage_id], mbar)
                    stage_id = (stage_id + 1) % num_stages
                    if stage_id == 0:
                        parity ^= 1

                # load V for PV MMA
                for i in cutlass.range_constexpr(BK // 16):
                    mbar = tma_full_mbar + stage_id
                    cute.arch.mbarrier_wait(tma_empty_mbar + stage_id, parity)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar, 16 * DIM * 2)
                    simple_tma_g2s(V_tma.atom, gV_tiles[None, iter_l * (BK // 16) + i], sV[None, None, stage_id], mbar)
                    stage_id = (stage_id + 1) % num_stages
                    if stage_id == 0:
                        parity ^= 1

        else:
            # MMA warps
            stage_id = 0
            parity = 0

            # ldmatrix.x4
            ldsm_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), BFloat16)
            ldsm_trans_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16)
            stsm_atom = cute.make_copy_atom(warp.StMatrix8x8x16bOp(num_matrices=4), BFloat16)

            assert WQ == 16
            sQ_ldsm = cute.zipped_divide(sQ, (16, cute.make_layout((8, 2, 4))))  # ((16,(8,2,4)), (BQ/16,DIM/64))
            sQ_ldsm = sQ_ldsm[(lane_id % 16, (None, lane_id // 16, None)), (warp_id % 2, None)]  # (8, 4, DIM/64)

            sK_ldsm = cute.zipped_divide(
                sK, (cute.make_layout((16, BK // 32)), cute.make_layout((8, 2)))
            )  # (((16,BK/32),(8,2)), (2,64/16,num_stages))
            sK_ldsm = sK_ldsm[
                (((lane_id // 16) * 8 + (lane_id % 8), None), (None, (lane_id // 8) % 2)), (warp_id // 2, None, None)
            ]  # (BK/32,8, 64/16,num_stages)

            sP_ldsm = cute.zipped_divide(sP, (cute.make_layout((8, 2, WK // 16)), 16))  # (((8,2,WK/16),16), (2,BQ/16))
            sP_ldsm = sP_ldsm[((None, lane_id // 16, None), lane_id % 16), None]  # (8,BK/32, (2,BQ/16))

            sV_ldsm = cute.zipped_divide(
                sV, (16, cute.make_layout((8, 2, DIM // 32)))
            )  # ((16,(8,2,DIM/32)), (1,2,num_stages))
            sV_ldsm = sV_ldsm[(lane_id % 16, (None, lane_id // 16, None)), (0, None, None)]  # (8,DIM/32, 2,num_stages)

            # sqrt(dim) / ln(2)
            sm_scale = cutlass.const_expr(DIM ** (-0.5) * 1.4426950408889634)

            rM = cute.make_rmem_tensor(2, Float32)
            rM.fill(float("-inf"))
            sumexp = cute.make_rmem_tensor(2, Float32)
            sumexp.fill(0.0)
            rO = cute.make_rmem_tensor((4, DIM // 16), Float32)
            rO.fill(0.0)

            # wait for Q TMA
            if warp_id == 0:
                cute.arch.mbarrier_wait(tma_q_mbar, 0)
            cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)

            for iter_l in range(cute.ceil_div(Lk, BK)):
                ##### QK MMA #####
                rQ = cute.make_rmem_tensor((8, 64 // 16), BFloat16)
                rK = cute.make_rmem_tensor(((4, 2), WK // 16), BFloat16)
                rS = cute.make_rmem_tensor((4, WK // 8), Float32)
                rS.fill(0.0)

                for i in cutlass.range_constexpr(DIM // 64):
                    if warp_id == 0:
                        cute.arch.mbarrier_wait(tma_full_mbar + stage_id, parity)
                    cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)

                    cute.copy(ldsm_atom, sQ_ldsm[None, None, i], rQ)
                    for k in cutlass.range_constexpr(64 // 16):
                        cute.copy(ldsm_atom, permute(sK_ldsm[None, None, k, stage_id], (1, 0)), rK)
                        for n in cutlass.range_constexpr(WK // 8):
                            rS[None, n] = mma_sync(rQ[None, k], rK[(None, n % 2), n // 2], rS[None, n])

                    cute.arch.mbarrier_arrive(tma_empty_mbar + stage_id)
                    stage_id = (stage_id + 1) % num_stages
                    if stage_id == 0:
                        parity ^= 1

                ##### online softmax #####
                for n in cutlass.range_constexpr(WK // 8):
                    for i in cutlass.range_constexpr(4):
                        rS[i, n] *= sm_scale

                # new rowmax
                rM_new0 = rM[0]
                rM_new1 = rM[1]
                for n in cutlass.range_constexpr(WK // 8):
                    rM_new0 = cute.arch.fmax(rM_new0, cute.arch.fmax(rS[0, n], rS[1, n]))
                    rM_new1 = cute.arch.fmax(rM_new1, cute.arch.fmax(rS[2, n], rS[3, n]))

                # butterfly reduction within 4 threads
                for i in cutlass.range_constexpr(2):
                    other0 = cute.arch.shuffle_sync_bfly(rM_new0, 1 << i)
                    other1 = cute.arch.shuffle_sync_bfly(rM_new1, 1 << i)
                    rM_new0 = cute.arch.fmax(rM_new0, other0)
                    rM_new1 = cute.arch.fmax(rM_new1, other1)

                # rowmax across 2 horizontal warps
                if lane_id % 4 == 0:
                    sM[0, lane_id // 4, warp_id] = rM_new0
                    sM[1, lane_id // 4, warp_id] = rM_new1
                cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)
                rM_new0 = cute.arch.fmax(rM_new0, sM[0, lane_id // 4, warp_id ^ 2])
                rM_new1 = cute.arch.fmax(rM_new1, sM[1, lane_id // 4, warp_id ^ 2])

                # rescale previous O
                rescale0 = cute.exp2(rM[0] - rM_new0, fastmath=True)
                rescale1 = cute.exp2(rM[1] - rM_new1, fastmath=True)
                for n in cutlass.range_constexpr(DIM // 16):
                    rO[0, n] *= rescale0
                    rO[1, n] *= rescale0
                    rO[2, n] *= rescale1
                    rO[3, n] *= rescale1

                # save the new rowmax
                rM[0] = rM_new0
                rM[1] = rM_new1

                # rowsumexp
                sumexp_new0 = Float32(0.0)
                sumexp_new1 = Float32(0.0)

                for n in cutlass.range_constexpr(WK // 8):
                    rS[0, n] = cute.exp2(rS[0, n] - rM[0], fastmath=True)
                    rS[1, n] = cute.exp2(rS[1, n] - rM[0], fastmath=True)
                    rS[2, n] = cute.exp2(rS[2, n] - rM[1], fastmath=True)
                    rS[3, n] = cute.exp2(rS[3, n] - rM[1], fastmath=True)
                    sumexp_new0 += rS[0, n] + rS[1, n]
                    sumexp_new1 += rS[2, n] + rS[3, n]

                # butterfly reduction within 4 threads
                for i in cutlass.range_constexpr(2):
                    sumexp_new0 += cute.arch.shuffle_sync_bfly(sumexp_new0, 1 << i)
                    sumexp_new1 += cute.arch.shuffle_sync_bfly(sumexp_new1, 1 << i)
                sumexp[0] = sumexp[0] * rescale0 + sumexp_new0
                sumexp[1] = sumexp[1] * rescale1 + sumexp_new1

                # pack S FP32 to P BF16
                rP = cute.make_rmem_tensor((8, WK // 16), BFloat16)
                rP.store(rS.load().to(BFloat16))
                cute.copy(stsm_atom, rP, sP_ldsm[None, None, (warp_id // 2, warp_id % 2)])
                cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)

                ##### PV MMA #####
                for j in cutlass.range_constexpr(2):
                    cute.copy(ldsm_atom, sP_ldsm[None, None, (j, warp_id % 2)], rP)

                    for i in cutlass.range_constexpr(WK // 16):
                        if warp_id == 0:
                            cute.arch.mbarrier_wait(tma_full_mbar + stage_id, parity)
                        cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)

                        rV = cute.make_rmem_tensor(((4, 2), DIM // 32), BFloat16)
                        cute.copy(ldsm_trans_atom, sV_ldsm[None, None, warp_id // 2, stage_id], rV)
                        for n in cutlass.range_constexpr(DIM // 16):
                            rO[None, n] = mma_sync(rP[None, i], rV[(None, n % 2), n // 2], rO[None, n])

                        cute.arch.mbarrier_arrive(tma_empty_mbar + stage_id)
                        stage_id = (stage_id + 1) % num_stages
                        if stage_id == 0:
                            parity ^= 1

            # sumexp across 2 horizontal warps
            if lane_id % 4 == 0:
                sM[0, lane_id // 4, warp_id] = sumexp[0]
                sM[1, lane_id // 4, warp_id] = sumexp[1]
            cute.arch.barrier(barrier_id=BAR_MMA, number_of_threads=128)

            sumexp[0] += sM[0, lane_id // 4, warp_id ^ 2]
            sumexp[1] += sM[1, lane_id // 4, warp_id ^ 2]
            sumexp[0] = cute.arch.rcp_approx(sumexp[0])
            sumexp[1] = cute.arch.rcp_approx(sumexp[1])

            # epilogue stores
            gO_view = cute.local_tile(
                gO[batch_id, None, None], (WQ, DIM // 2), (q_tile_id * 2 + warp_id % 2, warp_id // 2)
            )
            gO_view = permute(gO_view, (1, 0))  # (DIM/2, WQ)
            gO_view = cute.zipped_divide(
                gO_view, (cute.make_layout((2, 4)), cute.make_layout((8, 2)))
            )  # (((2,4),(8,2)), (DIM/16,1))
            gO_view = gO_view[((None, lane_id % 4), (lane_id // 4, None)), None]  # (2,2, (DIM/16,1))
            gO_view = cute.group_modes(gO_view, 0, 2)  # ((2,2), (DIM/8,1))

            st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32)
            for n in cutlass.range_constexpr(DIM // 16):
                rO[0, n] *= sumexp[0]
                rO[1, n] *= sumexp[0]
                rO[2, n] *= sumexp[1]
                rO[3, n] *= sumexp[1]

                tmp = cute.make_rmem_tensor(4, BFloat16)
                tmp.store(rO[None, n].load().to(BFloat16))
                cute.copy(st_atom, tmp, gO_view[None, n])

    @cache
    @staticmethod
    def compile():
        B = cute.sym_int()
        Lq = cute.sym_int()
        Lk = cute.sym_int()
        D = VaeAttn.DIM

        Q = make_fake_tensor(BFloat16, (B, Lq, D), (cute.sym_int64(16), D, 1), assumed_align=16)
        K = make_fake_tensor(BFloat16, (B, Lk, D), (cute.sym_int64(16), D, 1), assumed_align=16)
        V = make_fake_tensor(BFloat16, (B, Lk, D), (cute.sym_int64(16), D, 1), assumed_align=16)
        O = make_fake_tensor(BFloat16, (B, Lq, D), (cute.sym_int64(16), D, 1), assumed_align=16)

        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(VaeAttn(), Q, K, V, O, stream, options="--enable-tvm-ffi")


def vae_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if any(x.requires_grad for x in (q, k, v)):
        # F.sdpa's memory-efficient impl requires ndim=4
        return F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)

    o = torch.empty_like(q)
    VaeAttn.compile()(q, k, v, o)
    return o
