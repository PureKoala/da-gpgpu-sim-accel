// Copyright (c) 2026, GPGPU-Sim Research Team
// All rights reserved.
//
// Deformable Attention Function Model for GPGPU-Sim
// This is a functional model with fixed latency simulation

#ifndef DEFORM_ATTN_UNIT_H
#define DEFORM_ATTN_UNIT_H

#include "../abstract_hardware_model.h"
#include <vector>
#include <queue>

// Forward declarations
class ptx_instruction;
class ptx_thread_info;
class memory_space;

// ============================================================================
// Deformable Attention Request Structure
// ============================================================================

struct deform_attn_request {
    unsigned query_tile_id;        // Query Tile ID
    unsigned num_queries;          // 典型 16
    unsigned num_points_per_query; // 典型 16

    // PCB 数据
    float* weights;                // 16×16 FP16
    float threshold;
    bool enable_prune;

    // TBC 数据
    float* abs_coords;             // 16×16 × 2D
    bool* valid_coords;            // 16×16

    // TMA 数据
    addr_t row_base_addr;
    unsigned tile_x, tile_y;
    unsigned pitch;
    unsigned mode;                 // 0:Horizontal, 1:Vertical, 2:XOR, 3:Discrete
    unsigned tile_size;            // 0:16×16, 1:16×32, 2:32×16

    // 输出
    float* output;                 // 16×16 FP16

    deform_attn_request() {
        query_tile_id = 0;
        num_queries = 16;
        num_points_per_query = 16;
        weights = NULL;
        threshold = 0.5f;
        enable_prune = true;
        abs_coords = NULL;
        valid_coords = NULL;
        row_base_addr = 0;
        tile_x = 0;
        tile_y = 0;
        pitch = 0;
        mode = 3;  // 默认 Discrete
        tile_size = 0;
        output = NULL;
    }
};

// ============================================================================
// PCB (Pre-Check Block) - 权重预筛选
// ============================================================================

class deform_pcb_model {
public:
    deform_pcb_model();
    ~deform_pcb_model();

    // 功能：权重过滤，生成稀疏掩码
    // 延迟：1 cycle（固定）
    void execute(const float* weights, float threshold, bool enable, bool* mask);

    // 获取延迟
    unsigned get_latency() const { return 1; }

private:
    // 内部状态
    unsigned m_cycle_count;
};

// ============================================================================
// GTC (Gated Tensor Core) - 稀疏感知计算
// ============================================================================

class deform_tbc_model;

class deform_gtc_model {
public:
    deform_gtc_model();
    ~deform_gtc_model();

    // 功能：操作数隔离，标记有效坐标
    // 延迟：0 cycle（与 PCB 并行）
    void execute(const bool* mask, const float* coords, bool* valid_coords);

    // 获取延迟
    unsigned get_latency() const { return 0; }

private:
    // 内部状态
    unsigned m_cycle_count;
};

// ============================================================================
// TBC (Tile Boundary Check) - 自适应聚合决策
// ============================================================================

class deform_tbc_model {
public:
    enum mode_t {
        HORIZONTAL = 0,
        VERTICAL = 1,
        XOR = 2,
        DISCRETE = 3
    };

    enum tile_size_t {
        TILE_16x16 = 0,
        TILE_16x32 = 1,
        TILE_32x16 = 2
    };

    deform_tbc_model();
    ~deform_tbc_model();

    // Phase 1: 快速聚集判断
    // 延迟：4 cycles
    bool phase1_execute(const float* abs_coords, const bool* valid_coords,
                        mode_t& mode);

    // Phase 2: 精细模式分析
    // 延迟：+3 cycles（仅在 Phase 1 通过后执行）
    void phase2_execute(const float* abs_coords, const bool* valid_coords,
                        tile_size_t& tile_size, unsigned& base_tile_x,
                        unsigned& base_tile_y);

    // 完整执行（自动判断是否执行 Phase 2）
    // 延迟：4 cycles（低聚集度）或 7 cycles（高聚集度）
    bool execute(const float* abs_coords, const bool* valid_coords,
                 mode_t& mode, tile_size_t& tile_size,
                 unsigned& base_tile_x, unsigned& base_tile_y);

    // 获取延迟
    unsigned get_latency(bool high_gather_mode) const {
        return high_gather_mode ? 7 : 4;
    }

private:
    // 内部状态
    unsigned m_cycle_count;

    // 辅助函数
    bool check_boundary(float x, float y, int width, int height);
    unsigned calculate_tile_hit_rate(const float* abs_coords,
                                     const bool* valid_coords,
                                     unsigned base_tile_x,
                                     unsigned base_tile_y);
};

// ============================================================================
// TMA (Tile Memory Access) - Tile 加载
// ============================================================================

class deform_tma_model {
public:
    deform_tma_model();
    ~deform_tma_model();

    // 功能：Tile 加载（模拟内存访问）
    // 延迟：16 cycles（16 行 × 1 cycle）
    // 输入：base_tile_x, base_tile_y, tile_size, mode
    // 输出：tile_data（存储到 Shared Memory）
    void execute(addr_t gmem_base, addr_t smem_base,
                 unsigned tile_x, unsigned tile_y,
                 unsigned pitch, unsigned mode, unsigned tile_size,
                 memory_space* gmem, memory_space* smem,
                 unsigned smid, ptx_thread_info* thread,
                 const ptx_instruction* pI);

    // 获取延迟
    unsigned get_latency() const { return 16; }

private:
    // 内部状态
    unsigned m_cycle_count;

    // 辅助函数
    void apply_swizzle(float* data, unsigned mode);
};

// ============================================================================
// Storage - Tracker 管理 + Bank 映射
// ============================================================================

class deform_storage_model {
public:
    deform_storage_model();
    ~deform_storage_model();

    // 功能：Tracker 管理 + 自适应 Bank 映射
    // 延迟：6 cycles
    // 输入：Warp 查询（32 线程 × 2×2 像素）
    // 输出：2×2 邻域数据
    void execute(unsigned warp_id, unsigned query_id,
                 const float* tile_data, float* output,
                 unsigned mode, unsigned tile_width, unsigned tile_height);

    // 获取延迟
    unsigned get_latency() const { return 6; }

private:
    // 内部状态
    unsigned m_cycle_count;

    // CAM 管理（使用哈希表简化）
    struct tile_tracker {
        unsigned tile_id;
        unsigned row_bitmap;  // 每位表示一行是否加载完成
        bool ready;
    };

    std::vector<tile_tracker> m_tracker_table;

    // 辅助函数
    unsigned hash_tile_id(unsigned tile_id);
    unsigned calculate_bank_id(unsigned word_offset, unsigned mode);
};

// ============================================================================
// Interpolation - 双线性插值
// ============================================================================

class deform_interp_model {
public:
    deform_interp_model();
    ~deform_interp_model();

    // 功能：双线性插值计算
    // 延迟：3 cycles
    // 输入：p00-p11, wx, wy
    // 输出：result
    void execute(const float* p00, const float* p01,
                 const float* p10, const float* p11,
                 const float* wx, const float* wy,
                 float* result);

    // 获取延迟
    unsigned get_latency() const { return 3; }

private:
    // 内部状态
    unsigned m_cycle_count;
};

// ============================================================================
// Deformable Attention Unit (Top-Level)
// ============================================================================

class deform_attn_unit {
public:
    deform_attn_unit();
    ~deform_attn_unit();

    // 发起请求
    void issue(deform_attn_request* req);

    // 每周期调用（模拟流水线）
    void cycle();

    // 检查是否空闲
    bool is_idle() const;

    // 收集性能统计
    void collect_stats();

    // 重置统计
    void reset_stats();

private:
    // 子单元
    deform_pcb_model m_pcb;
    deform_gtc_model m_gtc;
    deform_tbc_model m_tbc;
    deform_tma_model m_tma;
    deform_storage_model m_storage;
    deform_interp_model m_interp;

    // 请求队列
    std::queue<deform_attn_request*> m_request_queue;

    // 当前处理状态
    enum stage_t {
        IDLE,
        PCB,
        GTC,
        TBC_PHASE1,
        TBC_PHASE2,
        TMA,
        STORAGE,
        INTERP,
        COMPLETE
    };

    stage_t m_current_stage;
    unsigned m_remaining_cycles;

    // 当前请求
    deform_attn_request* m_current_request;

    // 中间结果
    bool m_mask[16][16];
    bool m_valid_coords[16][16];
    deform_tbc_model::mode_t m_mode;
    deform_tbc_model::tile_size_t m_tile_size;
    unsigned m_base_tile_x;
    unsigned m_base_tile_y;
    float m_tile_data[16][16];
    float m_output[16][16];

    // 性能统计
    struct stats_t {
        unsigned num_ops;
        unsigned num_cycles;
        unsigned num_sparse_skip;
        unsigned num_discrete_fallback;
        unsigned num_tile_loads;
        unsigned num_high_gather;
        unsigned num_low_gather;
    };

    stats_t m_stats;

    // 辅助函数
    void process_pcb();
    void process_gtc();
    void process_tbc_phase1();
    void process_tbc_phase2();
    void process_tma();
    void process_storage();
    void process_interp();
    void complete_request();
};

#endif // DEFORM_ATTN_UNIT_H