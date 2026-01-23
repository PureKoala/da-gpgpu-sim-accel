// Copyright (c) 2026, GPGPU-Sim Research Team
// All rights reserved.
//
// Deformable Attention Function Model for GPGPU-Sim
// This is a functional model with fixed latency simulation

#include "deform_attn_unit.h"
#include "ptx_ir.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// ============================================================================
// PCB (Pre-Check Block) - 权重预筛选
// ============================================================================

deform_pcb_model::deform_pcb_model() : m_cycle_count(0) {}

deform_pcb_model::~deform_pcb_model() {}

void deform_pcb_model::execute(const float* weights, float threshold,
                               bool enable, bool* mask) {
    m_cycle_count++;

    if (!enable) {
        // 未启用剪枝，所有点都有效
        for (int i = 0; i < 16 * 16; i++) {
            mask[i] = false;  // 0 = 计算（有效）
        }
        return;
    }

    // 权重阈值比较
    for (int i = 0; i < 16 * 16; i++) {
        float abs_weight = std::fabs(weights[i]);
        mask[i] = (abs_weight < threshold);  // true = 跳过，false = 计算
    }
}

// ============================================================================
// GTC (Gated Tensor Core) - 稀疏感知计算
// ============================================================================

deform_gtc_model::deform_gtc_model() : m_cycle_count(0) {}

deform_gtc_model::~deform_gtc_model() {}

void deform_gtc_model::execute(const bool* mask, const float* coords,
                               bool* valid_coords) {
    m_cycle_count++;

    // 操作数隔离：根据掩码标记有效坐标
    for (int i = 0; i < 16 * 16; i++) {
        if (!mask[i]) {  // 有效点（mask=false）
            valid_coords[i] = true;
        } else {  // 无效点（mask=true）
            valid_coords[i] = false;
            // 操作数隔离：坐标置零（可选）
            // coords[i * 2] = 0.0f;     // x
            // coords[i * 2 + 1] = 0.0f; // y
        }
    }
}

// ============================================================================
// TBC (Tile Boundary Check) - 自适应聚合决策
// ============================================================================

deform_tbc_model::deform_tbc_model() : m_cycle_count(0) {}

deform_tbc_model::~deform_tbc_model() {}

bool deform_tbc_model::check_boundary(float x, float y, int width, int height) {
    // 边界检查：坐标必须在 [0, width) 和 [0, height) 范围内
    return (x >= 0 && x < width && y >= 0 && y < height);
}

unsigned deform_tbc_model::calculate_tile_hit_rate(const float* abs_coords,
                                                   const bool* valid_coords,
                                                   unsigned base_tile_x,
                                                   unsigned base_tile_y) {
    unsigned valid_count = 0;
    unsigned hit_count = 0;

    // 统计前 8 个有效点
    for (int i = 0; i < 16 * 16 && valid_count < 8; i++) {
        if (valid_coords[i]) {
            valid_count++;

            float x = abs_coords[i * 2];      // x 坐标
            float y = abs_coords[i * 2 + 1];  // y 坐标

            // 计算所在的 Tile
            unsigned tile_x = static_cast<unsigned>(x) / 16;
            unsigned tile_y = static_cast<unsigned>(y) / 16;

            if (tile_x == base_tile_x && tile_y == base_tile_y) {
                hit_count++;
            }
        }
    }

    if (valid_count == 0) {
        return 0;
    }

    return (hit_count * 100) / valid_count;  // 返回百分比
}

bool deform_tbc_model::phase1_execute(const float* abs_coords,
                                      const bool* valid_coords,
                                      mode_t& mode) {
    m_cycle_count += 4;  // Phase 1 固定 4 cycles

    // Step 1: 边界检查
    unsigned valid_count = 0;
    for (int i = 0; i < 16 * 16; i++) {
        if (valid_coords[i]) {
            float x = abs_coords[i * 2];
            float y = abs_coords[i * 2 + 1];

            // 边界检查（假设特征图尺寸为 32×32）
            if (!check_boundary(x, y, 32, 32)) {
                // 坐标越界，标记为无效
                // 注意：这里不修改 valid_coords，只是逻辑判断
                continue;
            }
            valid_count++;
        }
    }

    if (valid_count == 0) {
        mode = DISCRETE;
        return false;  // 无有效点，回退离散加载
    }

    // Step 2: 取第 1 个有效点作为锚点
    unsigned anchor_idx = 0;
    for (int i = 0; i < 16 * 16; i++) {
        if (valid_coords[i]) {
            anchor_idx = i;
            break;
        }
    }

    float anchor_x = abs_coords[anchor_idx * 2];
    float anchor_y = abs_coords[anchor_idx * 2 + 1];
    unsigned base_tile_x = static_cast<unsigned>(anchor_x) / 16;
    unsigned base_tile_y = static_cast<unsigned>(anchor_y) / 16;

    // Step 3: First-8 Voting
    unsigned hit_rate = calculate_tile_hit_rate(abs_coords, valid_coords,
                                                base_tile_x, base_tile_y);

    if (hit_rate < 50) {
        mode = DISCRETE;
        return false;  // 低聚集度，回退离散加载
    }

    // 高聚集度，进入 Phase 2
    return true;
}

void deform_tbc_model::phase2_execute(const float* abs_coords,
                                      const bool* valid_coords,
                                      tile_size_t& tile_size,
                                      unsigned& base_tile_x,
                                      unsigned& base_tile_y) {
    m_cycle_count += 3;  // Phase 2 固定 3 cycles

    // Step 1: 计算前 8 个有效点的包围盒
    float min_x = 1000000.0f, max_x = -1000000.0f;
    float min_y = 1000000.0f, max_y = -1000000.0f;
    unsigned valid_count = 0;

    for (int i = 0; i < 16 * 16 && valid_count < 8; i++) {
        if (valid_coords[i]) {
            float x = abs_coords[i * 2];
            float y = abs_coords[i * 2 + 1];

            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);

            valid_count++;
        }
    }

    if (valid_count == 0) {
        tile_size = TILE_16x16;
        return;
    }

    // Step 2: 计算 delta
    float delta_x = max_x - min_x;
    float delta_y = max_y - min_y;

    // Step 3: Tile 尺寸选择
    if (delta_x <= 16 && delta_y <= 16) {
        tile_size = TILE_16x16;
    } else if (delta_x <= 16 && delta_y > 16) {
        tile_size = TILE_16x32;
    } else if (delta_x > 16 && delta_y <= 16) {
        tile_size = TILE_32x16;
    } else {
        // 回退离散加载（已经在 Phase 1 处理）
        tile_size = TILE_16x16;
    }

    // Step 4: 确定锚点 Tile
    float anchor_x = min_x;
    float anchor_y = min_y;
    base_tile_x = static_cast<unsigned>(anchor_x) / 16;
    base_tile_y = static_cast<unsigned>(anchor_y) / 16;
}

bool deform_tbc_model::execute(const float* abs_coords, const bool* valid_coords,
                               mode_t& mode, tile_size_t& tile_size,
                               unsigned& base_tile_x, unsigned& base_tile_y) {
    // Phase 1: 快速聚集判断
    bool high_gather_mode = phase1_execute(abs_coords, valid_coords, mode);

    if (!high_gather_mode) {
        // 低聚集度，直接返回
        tile_size = TILE_16x16;
        base_tile_x = 0;
        base_tile_y = 0;
        return false;
    }

    // Phase 2: 精细模式分析
    phase2_execute(abs_coords, valid_coords, tile_size, base_tile_x, base_tile_y);

    // 确定最终模式
    float delta_x = 0, delta_y = 0;  // 需要重新计算（简化）
    // 实际实现中需要保存 Phase 2 的计算结果

    // 简化：根据 tile_size 判断模式
    if (tile_size == TILE_16x16) {
        mode = XOR;  // 默认 XOR 模式
    } else if (tile_size == TILE_16x32) {
        mode = HORIZONTAL;
    } else if (tile_size == TILE_32x16) {
        mode = VERTICAL;
    }

    return true;
}

// ============================================================================
// TMA (Tile Memory Access) - Tile 加载
// ============================================================================

deform_tma_model::deform_tma_model() : m_cycle_count(0) {}

deform_tma_model::~deform_tma_model() {}

void deform_tma_model::apply_swizzle(float* data, unsigned mode) {
    // 应用布局变换（逻辑上，实际由硬件处理）
    // 这里只是占位，实际在 Storage 模块处理 Bank 映射
}

void deform_tma_model::execute(addr_t gmem_base, addr_t smem_base,
                               unsigned tile_x, unsigned tile_y,
                               unsigned pitch, unsigned mode, unsigned tile_size,
                               memory_space* gmem, memory_space* smem,
                               unsigned smid, ptx_thread_info* thread,
                               const ptx_instruction* pI) {
    m_cycle_count += 16;  // 固定 16 cycles

    // 确定 Tile 尺寸
    unsigned tile_width, tile_height;
    switch (tile_size) {
        case 0: tile_width = 16; tile_height = 16; break;
        case 1: tile_width = 16; tile_height = 32; break;
        case 2: tile_width = 32; tile_height = 16; break;
        default: tile_width = 16; tile_height = 16; break;
    }

    // 每行加载 32 bytes（16 个 FP16）
    unsigned bytes_per_row = tile_width * 2;  // FP16 = 2 bytes

    // Step 1: 从 GMEM 加载数据
    for (unsigned row = 0; row < tile_height; row++) {
        addr_t gmem_addr = gmem_base + (tile_y * pitch + tile_x) * 2 + row * pitch * 2;

        // 读取一行数据
        uint16_t row_data[32];  // 最多 32 个 FP16
        unsigned elements_to_read = std::min(tile_width, 32u);

        for (unsigned col = 0; col < elements_to_read; col++) {
            gmem->read(gmem_addr + col * 2, sizeof(uint16_t), &row_data[col]);
        }

        // Step 2: 写入 SMEM（逻辑地址连续）
        addr_t smem_addr = smem_base + (row * tile_width) * 2;

        for (unsigned col = 0; col < elements_to_read; col++) {
            smem->write(smem_addr + col * 2, sizeof(uint16_t), &row_data[col], thread, pI);
        }
    }

    // Step 3: 应用布局变换（逻辑上）
    apply_swizzle(NULL, mode);
}

// ============================================================================
// Storage - Tracker 管理 + Bank 映射
// ============================================================================

deform_storage_model::deform_storage_model() : m_cycle_count(0) {
    // 初始化 Tracker 表（32 个条目）
    m_tracker_table.resize(32);
    for (unsigned i = 0; i < 32; i++) {
        m_tracker_table[i].tile_id = 0;
        m_tracker_table[i].row_bitmap = 0;
        m_tracker_table[i].ready = false;
    }
}

deform_storage_model::~deform_storage_model() {}

unsigned deform_storage_model::hash_tile_id(unsigned tile_id) {
    // 简单的哈希函数
    return tile_id % 32;
}

unsigned deform_storage_model::calculate_bank_id(unsigned word_offset, unsigned mode) {
    // 自适应 Bank 映射
    unsigned xor_key = 0;

    switch (mode) {
        case 0:  // Horizontal
            xor_key = (word_offset >> 2) & 0x1F;  // y高位
            break;
        case 1:  // Vertical
            xor_key = ((word_offset >> 1) >> 2) & 0x1F;  // x高位
            break;
        case 2:  // XOR
            xor_key = ((word_offset >> 2) ^ (word_offset & 0x1F)) & 0x1F;
            break;
        default:
            xor_key = 0;
            break;
    }

    return (word_offset ^ xor_key) % 32;
}

void deform_storage_model::execute(unsigned warp_id, unsigned query_id,
                                   const float* tile_data, float* output,
                                   unsigned mode, unsigned tile_width,
                                   unsigned tile_height) {
    m_cycle_count += 6;  // 固定 6 cycles

    // 模拟 CAM 查找（使用哈希表）
    unsigned tile_id = warp_id * 1000 + query_id;
    unsigned hash_idx = hash_tile_id(tile_id);

    // 检查 Tracker
    if (!m_tracker_table[hash_idx].ready) {
        // Tile 未加载，等待（简化：假设已加载）
        m_tracker_table[hash_idx].tile_id = tile_id;
        m_tracker_table[hash_idx].row_bitmap = 0xFFFFFFFF;  // 所有行加载完成
        m_tracker_table[hash_idx].ready = true;
    }

    // Step 1: 读取 2×2 邻域数据（32 线程 × 2×2 像素）
    for (unsigned tid = 0; tid < 32; tid++) {
        // 每个线程处理 2×2 像素
        unsigned base_idx = tid * 4;  // 4 个像素

        // 计算 4 个像素的坐标（简化）
        unsigned row = (tid / 4) * 2;
        unsigned col = (tid % 4) * 2;

        // 读取 4 个像素（p00, p01, p10, p11）
        for (unsigned p = 0; p < 4; p++) {
            unsigned r = row + (p / 2);
            unsigned c = col + (p % 2);

            if (r < tile_height && c < tile_width) {
                unsigned idx = r * tile_width + c;

                // 自适应 Bank 映射
                unsigned word_offset = idx;
                unsigned bank_id = calculate_bank_id(word_offset, mode);

                // 读取数据（这里只是模拟，实际由硬件处理）
                output[base_idx + p] = tile_data[idx];
            }
        }
    }
}

// ============================================================================
// Interpolation - 双线性插值
// ============================================================================

deform_interp_model::deform_interp_model() : m_cycle_count(0) {}

deform_interp_model::~deform_interp_model() {}

void deform_interp_model::execute(const float* p00, const float* p01,
                                  const float* p10, const float* p11,
                                  const float* wx, const float* wy,
                                  float* result) {
    m_cycle_count += 3;  // 固定 3 cycles

    // 双线性插值公式：
    // result = p00*(1-wx)*(1-wy) + p01*wx*(1-wy) + p10*(1-wx)*wy + p11*wx*wy

    for (int i = 0; i < 16 * 16; i++) {
        float w_x = wx[i];
        float w_y = wy[i];

        float term00 = p00[i] * (1.0f - w_x) * (1.0f - w_y);
        float term01 = p01[i] * w_x * (1.0f - w_y);
        float term10 = p10[i] * (1.0f - w_x) * w_y;
        float term11 = p11[i] * w_x * w_y;

        result[i] = term00 + term01 + term10 + term11;
    }
}

// ============================================================================
// Deformable Attention Unit (Top-Level)
// ============================================================================

deform_attn_unit::deform_attn_unit()
    : m_current_stage(IDLE), m_remaining_cycles(0), m_current_request(NULL) {
    reset_stats();
}

deform_attn_unit::~deform_attn_unit() {
    // 清空请求队列
    while (!m_request_queue.empty()) {
        deform_attn_request* req = m_request_queue.front();
        m_request_queue.pop();
        delete req;
    }
}

void deform_attn_unit::issue(deform_attn_request* req) {
    m_request_queue.push(req);
}

void deform_attn_unit::cycle() {
    if (m_current_stage == IDLE) {
        if (m_request_queue.empty()) {
            return;  // 无请求
        }

        // 取出下一个请求
        m_current_request = m_request_queue.front();
        m_request_queue.pop();

        // 开始处理
        m_current_stage = PCB;
        m_remaining_cycles = m_pcb.get_latency();
        m_stats.num_ops++;
    }

    // 减少剩余周期数
    if (m_remaining_cycles > 0) {
        m_remaining_cycles--;
        m_stats.num_cycles++;

        if (m_remaining_cycles == 0) {
            // 当前阶段完成，进入下一阶段
            switch (m_current_stage) {
                case PCB:
                    process_pcb();
                    break;
                case GTC:
                    process_gtc();
                    break;
                case TBC_PHASE1:
                    process_tbc_phase1();
                    break;
                case TBC_PHASE2:
                    process_tbc_phase2();
                    break;
                case TMA:
                    process_tma();
                    break;
                case STORAGE:
                    process_storage();
                    break;
                case INTERP:
                    process_interp();
                    break;
                default:
                    break;
            }
        }
    }
}

void deform_attn_unit::process_pcb() {
    // 执行 PCB
    m_pcb.execute(m_current_request->weights,
                  m_current_request->threshold,
                  m_current_request->enable_prune,
                  &m_mask[0][0]);

    // 进入 GTC（与 PCB 并行，延迟 0）
    m_current_stage = GTC;
    m_remaining_cycles = m_gtc.get_latency();
}

void deform_attn_unit::process_gtc() {
    // 执行 GTC
    m_gtc.execute(&m_mask[0][0],
                  m_current_request->abs_coords,
                  &m_valid_coords[0][0]);

    // 进入 TBC Phase 1
    m_current_stage = TBC_PHASE1;
    m_remaining_cycles = m_tbc.get_latency(false);  // Phase 1 固定 4 cycles
}

void deform_attn_unit::process_tbc_phase1() {
    // 执行 TBC Phase 1
    bool high_gather_mode = m_tbc.phase1_execute(m_current_request->abs_coords,
                                           &m_valid_coords[0][0],
                                           m_mode);

    if (!high_gather_mode) {
        // 低聚集度，回退离散加载
        m_stats.num_discrete_fallback++;
        m_stats.num_low_gather++;

        // 直接进入 TMA（Discrete 模式）
        m_mode = deform_tbc_model::DISCRETE;
        m_tile_size = deform_tbc_model::TILE_16x16;
        m_base_tile_x = 0;
        m_base_tile_y = 0;

        m_current_stage = TMA;
        m_remaining_cycles = m_tma.get_latency();
    } else {
        // 高聚集度，进入 Phase 2
        m_stats.num_high_gather++;

        m_current_stage = TBC_PHASE2;
        m_remaining_cycles = m_tbc.get_latency(true) - 4;  // Phase 2 额外 3 cycles
    }
}

void deform_attn_unit::process_tbc_phase2() {
    // 执行 TBC Phase 2
    m_tbc.phase2_execute(m_current_request->abs_coords,
                         &m_valid_coords[0][0],
                         m_tile_size,
                         m_base_tile_x,
                         m_base_tile_y);

    // 进入 TMA
    m_current_stage = TMA;
    m_remaining_cycles = m_tma.get_latency();
}

void deform_attn_unit::process_tma() {
    // 执行 TMA（需要 memory_space，这里简化）
    // 实际实现中需要从 shader_core_ctx 获取 memory_space

    // 统计 Tile 加载
    m_stats.num_tile_loads++;

    // 进入 Storage
    m_current_stage = STORAGE;
    m_remaining_cycles = m_storage.get_latency();
}

void deform_attn_unit::process_storage() {
    // 执行 Storage
    m_storage.execute(m_current_request->query_tile_id,
                      0,  // query_id（简化）
                      &m_tile_data[0][0],
                      &m_output[0][0],
                      m_mode,
                      16,  // tile_width（简化）
                      16); // tile_height（简化）

    // 进入 Interpolation
    m_current_stage = INTERP;
    m_remaining_cycles = m_interp.get_latency();
}

void deform_attn_unit::process_interp() {
    // 执行 Interpolation（需要插值权重，这里简化）
    // 实际实现中需要从参数读取 wx, wy

    // 完成请求
    complete_request();
}

void deform_attn_unit::complete_request() {
    // 复制结果到输出
    if (m_current_request->output) {
        memcpy(m_current_request->output, m_output, sizeof(float) * 16 * 16);
    }

    // 清理当前请求
    delete m_current_request;
    m_current_request = NULL;

    // 回到空闲状态
    m_current_stage = IDLE;
    m_remaining_cycles = 0;
}

bool deform_attn_unit::is_idle() const {
    return m_current_stage == IDLE && m_request_queue.empty();
}

void deform_attn_unit::collect_stats() {
    printf("=== Deformable Attention Stats ===\n");
    printf("Total Operations: %u\n", m_stats.num_ops);
    printf("Total Cycles: %u\n", m_stats.num_cycles);
    printf("Sparse Skip Rate: %.1f%%\n",
           (m_stats.num_ops > 0) ? (m_stats.num_sparse_skip * 100.0f / m_stats.num_ops) : 0.0f);
    printf("Discrete Fallback Rate: %.1f%%\n",
           (m_stats.num_ops > 0) ? (m_stats.num_discrete_fallback * 100.0f / m_stats.num_ops) : 0.0f);
    printf("Tile Load Count: %u\n", m_stats.num_tile_loads);
    printf("High Gather Count: %u\n", m_stats.num_high_gather);
    printf("Low Gather Count: %u\n", m_stats.num_low_gather);
    printf("Average Latency: %.1f cycles\n",
           (m_stats.num_ops > 0) ? (m_stats.num_cycles * 1.0f / m_stats.num_ops) : 0.0f);
    printf("==================================\n");
}

void deform_attn_unit::reset_stats() {
    memset(&m_stats, 0, sizeof(stats_t));
}