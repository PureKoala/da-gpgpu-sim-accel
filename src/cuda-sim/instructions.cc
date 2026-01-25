// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Jimmy Kwa, George L. Yuan, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "instructions.h"
#include "half.h"
#include "half.hpp"
#include "opcodes.h"
#include "ptx_ir.h"
#include "ptx_sim.h"
typedef void *yyscan_t;
class ptx_recognizer;
#include <assert.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/shader.h"
#include "cuda-math.h"
#include "cuda_device_printf.h"
#include "ptx.tab.h"
#include "ptx_loader.h"

// Jin: include device runtime for CDP
#include "cuda_device_runtime.h"

#include <stdarg.h>
#include "../../libcuda/gpgpu_context.h"

using half_float::half;

namespace {
struct deform_block_key {
  unsigned kernel_uid;
  unsigned cta_x;
  unsigned cta_y;
  unsigned cta_z;
  unsigned level;
};

struct deform_block_state {
  int num_points = 0;
  int valid_points = 0;
  int mode = 0;
  int tile_w = 0;
  int tile_h = 0;
  bool pcb_latency_applied = false;
  bool tbc_latency_applied = false;
  bool tma_latency_applied = false;
  bool interp_latency_applied = false;
};

struct deform_block_key_hash {
  std::size_t operator()(const deform_block_key &k) const noexcept {
    std::size_t h = 1469598103934665603ull;
    auto mix = [&](std::size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix(k.kernel_uid);
    mix(k.cta_x);
    mix(k.cta_y);
    mix(k.cta_z);
    mix(k.level);
    return h;
  }
};

inline bool operator==(const deform_block_key &a,
                       const deform_block_key &b) noexcept {
  return a.kernel_uid == b.kernel_uid && a.cta_x == b.cta_x &&
         a.cta_y == b.cta_y && a.cta_z == b.cta_z && a.level == b.level;
}

static std::unordered_map<deform_block_key, deform_block_state,
                          deform_block_key_hash>
    g_deform_block_state;

inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

inline unsigned clamp_latency(float v) {
  if (v < 1.0f) return 1;
  // Keep within scheduler occupancy bitset bounds (MAX_ALU_LATENCY=512).
  if (v > 511.0f) return 511;
  return static_cast<unsigned>(v + 0.5f);
}

inline float level_scale_or_default(const shader_core_config *cfg, int level) {
  if (level < 0) return 1.0f;
  if (level > 3) level = 3;
  return cfg->deform_level_scale[level];
}

inline float sparsity_factor_or_default(const shader_core_config *cfg,
                                        int valid_points, int num_points) {
  if (num_points <= 0) return 1.0f;
  float valid_ratio = static_cast<float>(valid_points) /
                      static_cast<float>(num_points);
  float sparsity = 1.0f - clampf(valid_ratio, 0.0f, 1.0f);
  float factor = 1.0f - cfg->deform_sparsity_alpha * sparsity;
  return clampf(factor, cfg->deform_sparsity_min_factor,
                cfg->deform_sparsity_max_factor);
}

inline deform_block_state &get_block_state(kernel_info_t &k, dim3 ctaid,
                                           int level) {
  deform_block_key key;
  key.kernel_uid = k.get_uid();
  key.cta_x = ctaid.x;
  key.cta_y = ctaid.y;
  key.cta_z = ctaid.z;
  key.level = static_cast<unsigned>(level);
  return g_deform_block_state[key];
}
}  // namespace

const char *g_opcode_string[NUM_OPCODES] = {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) STR,
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) STR,
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
};
// Using profiled information::check the TensorCoreMatrixArrangement.xls for
// details
unsigned thread_group_offset(int thread, unsigned wmma_type,
                             unsigned wmma_layout, unsigned type, int stride) {
  unsigned offset;
  
  // INT8 WMMA: Each element is 1 byte (vs 2 bytes for FP16)
  // The offset values should be ELEMENT indices, not byte offsets
  // So INT8 uses the SAME offset tables as FP16 (element-wise addressing)
  // The byte offset calculation happens later: offset * (size/8)
  
  // FP16 WMMA: Original offset tables (m16n16k16.f16) - in ELEMENT indices
  unsigned load_a_row[8] = {0, 128, 0, 128, 64, 192, 64, 192};
  unsigned load_a_col[8] = {0, 8, 0, 8, 4, 12, 4, 12};
  unsigned load_b_row[8] = {0, 8, 0, 8, 4, 12, 4, 12};
  unsigned load_b_col[8] = {0, 128, 0, 128, 64, 192, 64, 192};
  unsigned load_c_float_row[8] = {0, 128, 8, 136, 64, 192, 72, 200};
  unsigned load_c_float_col[8] = {0, 8, 128, 136, 4, 12, 132, 140};
  unsigned load_c_half_row[8] = {0, 128, 8, 136, 64, 192, 72, 200};
  unsigned load_c_half_col[8] = {0, 8, 128, 136, 4, 12, 132, 140};
  unsigned thread_group = thread / 4;
  unsigned in_tg_index = thread % 4;
  
  // Detect if this is INT8 WMMA
  bool is_int8 = (type == S8_TYPE || type == U8_TYPE);

  switch (wmma_type) {
    case LOAD_A:
      // Use same offset table for both INT8 and FP16
      // The element indices are the same, byte offset differs due to size/8
      if (wmma_layout == ROW)
        offset = load_a_row[thread_group] + 16 * in_tg_index;
      else
        offset = load_a_col[thread_group] + 16 * in_tg_index;
      break;

    case LOAD_B:
      // Use same offset table for both INT8 and FP16
      if (wmma_layout == ROW)
        offset = load_b_row[thread_group] + 16 * in_tg_index;
      else
        offset = load_b_col[thread_group] + 16 * in_tg_index;
      break;

    case LOAD_C:
    case STORE_D:
      if (type == F16_TYPE) {
        if (wmma_layout == ROW)
          offset = load_c_half_row[thread_group] + 16 * in_tg_index;
        else
          offset = load_c_half_col[thread_group] + in_tg_index;
      } else if (type == S32_TYPE || type == F32_TYPE) {
        // INT32 or FP32 accumulator (used by INT8 WMMA output or FP16 WMMA output)
        if (wmma_layout == ROW)
          offset = load_c_float_row[thread_group];
        else
          offset = load_c_float_col[thread_group];

        switch (in_tg_index) {
          case 0:
            break;
          case 1:
            if (wmma_layout == ROW)
              offset += 16;
            else
              offset += 1;
            break;
          case 2:
            if (wmma_layout == ROW)
              offset += 2;
            else
              offset += 32;
            break;
          case 3:
            if (wmma_layout == ROW)
              offset += 18;
            else
              offset += 33;
            break;
          default:
            abort();
        }
      } else {
        printf("thread_group_offset: unsupported type=%u for LOAD_C/STORE_D\n", type);
        abort();
      }
      break;

    default:
      abort();
  }
  offset = (offset / 16) * stride + offset % 16;
  return offset;
}

int acc_float_offset(int index, int wmma_layout, int stride) {
  int c_row_offset[] = {0, 1, 32, 33, 4, 5, 36, 37};
  int c_col_offset[] = {0, 16, 2, 18, 64, 80, 66, 82};
  int offset;

  if (wmma_layout == ROW)
    offset = c_row_offset[index];
  else if (wmma_layout == COL)
    offset = c_col_offset[index];
  else {
    printf("wrong layout");
    abort();
  }
  offset = (offset / 16) * stride + offset % 16;
  return offset;
}

void inst_not_implemented(const ptx_instruction *pI);

// FMR (Feature Map Reorganizer) timing model implementation
void ld_sample_fmr_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);

// Deformable Attention Pipeline Implementation
void deform_pcb_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);
void deform_gtc_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);
void deform_tbc_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);
void deform_tma_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);
void deform_interp_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);

ptx_reg_t srcOperandModifiers(ptx_reg_t opData, operand_info opInfo,
                              operand_info dstInfo, unsigned type,
                              ptx_thread_info *thread);

void video_mem_instruction(const ptx_instruction *pI, ptx_thread_info *thread,
                           int op_code);

void sign_extend(ptx_reg_t &data, unsigned src_size, const operand_info &dst);

void ptx_thread_info::set_reg(const symbol *reg, const ptx_reg_t &value) {
  assert(reg != NULL);
  if (reg->name() == "_") return;
  assert(!m_regs.empty());
  assert(reg->uid() > 0);
  m_regs.back()[reg] = value;
  if (m_enable_debug_trace) m_debug_trace_regs_modified.back()[reg] = value;
  m_last_set_operand_value = value;
}

void ptx_thread_info::print_reg_thread(char *fname) {
  FILE *fp = fopen(fname, "w");
  assert(fp != NULL);

  int size = m_regs.size();

  if (size > 0) {
    reg_map_t reg = m_regs.back();

    reg_map_t::const_iterator it;
    for (it = reg.begin(); it != reg.end(); ++it) {
      const std::string &name = it->first->name();
      const std::string &dec = it->first->decl_location();
      unsigned size = it->first->get_size_in_bytes();
      fprintf(fp, "%s %llu %s %d\n", name.c_str(), it->second, dec.c_str(),
              size);
    }
    // m_regs.pop_back();
  }
  fclose(fp);
}

void ptx_thread_info::resume_reg_thread(char *fname, symbol_table *symtab) {
  FILE *fp2 = fopen(fname, "r");
  assert(fp2 != NULL);
  // m_regs.push_back( reg_map_t() );
  char line[200];
  while (fgets(line, sizeof line, fp2) != NULL) {
    symbol *reg;
    char *pch;
    pch = strtok(line, " ");
    char *name = pch;
    reg = symtab->lookup(name);
    ptx_reg_t data;
    pch = strtok(NULL, " ");
    data = atoi(pch);
    pch = strtok(NULL, " ");
    pch = strtok(NULL, " ");
    m_regs.back()[reg] = data;
  }
  fclose(fp2);
}

ptx_reg_t ptx_thread_info::get_reg(const symbol *reg) {
  static bool unfound_register_warned = false;
  assert(reg != NULL);
  assert(!m_regs.empty());
  reg_map_t::iterator regs_iter = m_regs.back().find(reg);
  if (regs_iter == m_regs.back().end()) {
    assert(reg->type()->get_key().is_reg());
    const std::string &name = reg->name();
    unsigned call_uid = m_callstack.back().m_call_uid;
    ptx_reg_t uninit_reg;
    uninit_reg.u32 = 0x0;
    set_reg(reg, uninit_reg);  // give it a value since we are going to warn the
                               // user anyway
    std::string file_loc = get_location();
    if (!unfound_register_warned) {
      printf(
          "GPGPU-Sim PTX: WARNING (%s) ** reading undefined register \'%s\' "
          "(cuid:%u). Setting to 0X00000000. This is okay if you are "
          "simulating the native ISA"
          "\n",
          file_loc.c_str(), name.c_str(), call_uid);
      unfound_register_warned = true;
    }
    regs_iter = m_regs.back().find(reg);
  }
  if (m_enable_debug_trace)
    m_debug_trace_regs_read.back()[reg] = regs_iter->second;
  return regs_iter->second;
}

ptx_reg_t ptx_thread_info::get_operand_value(const operand_info &op,
                                             operand_info dstInfo,
                                             unsigned opType,
                                             ptx_thread_info *thread,
                                             int derefFlag) {
  ptx_reg_t result, tmp;

  if (op.get_double_operand_type() == 0) {
    if (((opType != BB128_TYPE) && (opType != BB64_TYPE) &&
         (opType != FF64_TYPE)) ||
        (op.get_addr_space() != undefined_space)) {
      if (op.is_reg()) {
        result = get_reg(op.get_symbol());
      } else if (op.is_builtin()) {
        result.u32 = get_builtin(op.get_int(), op.get_addr_offset());
      } else if (op.is_immediate_address()) {
        result.u64 = op.get_addr_offset();
      } else if (op.is_memory_operand()) {
        // a few options here...
        const symbol *sym = op.get_symbol();
        const type_info *type = sym->type();
        const type_info_key &info = type->get_key();

        if (info.is_reg()) {
          const symbol *name = op.get_symbol();
          result.u64 = get_reg(name).u64 + op.get_addr_offset();
        } else if (info.is_param_kernel()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_param_local()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_global()) {
          assert(op.get_addr_offset() == 0);
          result.u64 = sym->get_address();
        } else if (info.is_local()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_const()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (op.is_shared()) {
          result.u64 = op.get_symbol()->get_address() + op.get_addr_offset();
        } else if (op.is_sstarr()) {
          result.u64 = op.get_symbol()->get_address() + op.get_addr_offset();
        } else {
          const char *name = op.name().c_str();
          printf(
              "GPGPU-Sim PTX: ERROR ** get_operand_value : unknown memory "
              "operand type for %s\n",
              name);
          abort();
        }

      } else if (op.is_literal()) {
        result = op.get_literal_value();
      } else if (op.is_label()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_shared()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_sstarr()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_const()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_global()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_local()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_function_address()) {
        result.u64 = (size_t)op.get_symbol()->get_pc();
      } else if (op.is_param_kernel()) {
        result.u64 = op.get_symbol()->get_address();
      } else {
        const char *name = op.name().c_str();
        const symbol *sym2 = op.get_symbol();
        const type_info *type2 = sym2->type();
        const type_info_key &info2 = type2->get_key();
        if (info2.is_param_kernel()) {
          result.u64 = sym2->get_address() + op.get_addr_offset();
        } else {
          printf(
              "GPGPU-Sim PTX: ERROR ** get_operand_value : unknown operand "
              "type for %s\n",
              name);
          assert(0);
        }
      }

      if (op.get_operand_lohi() == 1)
        result.u64 = result.u64 & 0xFFFF;
      else if (op.get_operand_lohi() == 2)
        result.u64 = (result.u64 >> 16) & 0xFFFF;
    } else if (opType == BB128_TYPE) {
      // b128
      result.u128.lowest = get_reg(op.vec_symbol(0)).u32;
      result.u128.low = get_reg(op.vec_symbol(1)).u32;
      result.u128.high = get_reg(op.vec_symbol(2)).u32;
      result.u128.highest = get_reg(op.vec_symbol(3)).u32;
    } else {
      // bb64 or ff64
      result.bits.ls = get_reg(op.vec_symbol(0)).u32;
      result.bits.ms = get_reg(op.vec_symbol(1)).u32;
    }
  } else if (op.get_double_operand_type() == 1) {
    ptx_reg_t firstHalf, secondHalf;
    firstHalf.u64 = get_reg(op.vec_symbol(0)).u64;
    secondHalf.u64 = get_reg(op.vec_symbol(1)).u64;
    if (op.get_operand_lohi() == 1)
      secondHalf.u64 = secondHalf.u64 & 0xFFFF;
    else if (op.get_operand_lohi() == 2)
      secondHalf.u64 = (secondHalf.u64 >> 16) & 0xFFFF;
    result.u64 = firstHalf.u64 + secondHalf.u64;
  } else if (op.get_double_operand_type() == 2) {
    // s[reg1 += reg2]
    // reg1 is incremented after value is returned: the value returned is
    // s[reg1]
    ptx_reg_t firstHalf, secondHalf;
    firstHalf.u64 = get_reg(op.vec_symbol(0)).u64;
    secondHalf.u64 = get_reg(op.vec_symbol(1)).u64;
    if (op.get_operand_lohi() == 1)
      secondHalf.u64 = secondHalf.u64 & 0xFFFF;
    else if (op.get_operand_lohi() == 2)
      secondHalf.u64 = (secondHalf.u64 >> 16) & 0xFFFF;
    result.u64 = firstHalf.u64;
    firstHalf.u64 = firstHalf.u64 + secondHalf.u64;
    set_reg(op.vec_symbol(0), firstHalf);
  } else if (op.get_double_operand_type() == 3) {
    // s[reg += immediate]
    // reg is incremented after value is returned: the value returned is s[reg]
    ptx_reg_t firstHalf;
    firstHalf.u64 = get_reg(op.get_symbol()).u64;
    result.u64 = firstHalf.u64;
    firstHalf.u64 = firstHalf.u64 + op.get_addr_offset();
    set_reg(op.get_symbol(), firstHalf);
  }

  ptx_reg_t finalResult;
  memory_space *mem = NULL;
  size_t size = 0;
  int t = 0;
  finalResult.u64 = 0;

  // complete other cases for reading from memory, such as reading from other
  // const memory
  if ((op.get_addr_space() == global_space) && (derefFlag)) {
    // global memory - g[4], g[$r0]
    mem = thread->get_global_memory();
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = global_space;

    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == shared_space) && (derefFlag)) {
    // shared memory - s[4], s[$r0]
    mem = thread->m_shared_mem;
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = shared_space;

    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == const_space) && (derefFlag)) {
    // const memory - ce0c1[4], ce0c1[$r0]
    mem = thread->get_global_memory();
    type_info_key::type_decode(opType, size, t);
    mem->read((result.u32 + op.get_const_mem_offset()), size / 8,
              &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = const_space;
    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == local_space) && (derefFlag)) {
    // local memory - l0[4], l0[$r0]
    mem = thread->m_local_mem;
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = local_space;
    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else {
    finalResult = result;
  }

  if ((op.get_operand_neg() == true) && (derefFlag)) {
    switch (opType) {
      // Default to f32 for now, need to add support for others
      case S8_TYPE:
      case U8_TYPE:
      case B8_TYPE:
        finalResult.s8 = -finalResult.s8;
        break;
      case S16_TYPE:
      case U16_TYPE:
      case B16_TYPE:
        finalResult.s16 = -finalResult.s16;
        break;
      case S32_TYPE:
      case U32_TYPE:
      case B32_TYPE:
        finalResult.s32 = -finalResult.s32;
        break;
      case S64_TYPE:
      case U64_TYPE:
      case B64_TYPE:
        finalResult.s64 = -finalResult.s64;
        break;
      case F16_TYPE:
        finalResult.f16 = -finalResult.f16;
        break;
      case F32_TYPE:
        finalResult.f32 = -finalResult.f32;
        break;
      case F64_TYPE:
      case FF64_TYPE:
        finalResult.f64 = -finalResult.f64;
        break;
      default:
        assert(0);
    }
  }

  return finalResult;
}

unsigned get_operand_nbits(const operand_info &op) {
  if (op.is_reg()) {
    const symbol *sym = op.get_symbol();
    const type_info *typ = sym->type();
    type_info_key t = typ->get_key();
    switch (t.scalar_type()) {
      case PRED_TYPE:
        return 1;
      case B8_TYPE:
      case S8_TYPE:
      case U8_TYPE:
        return 8;
      case S16_TYPE:
      case U16_TYPE:
      case F16_TYPE:
      case B16_TYPE:
        return 16;
      case S32_TYPE:
      case U32_TYPE:
      case F32_TYPE:
      case B32_TYPE:
        return 32;
      case S64_TYPE:
      case U64_TYPE:
      case F64_TYPE:
      case B64_TYPE:
        return 64;
      default:
        printf("ERROR: unknown register type\n");
        fflush(stdout);
        abort();
    }
  } else {
    printf(
        "ERROR: Need to implement get_operand_nbits() for currently "
        "unsupported operand_info type\n");
    fflush(stdout);
    abort();
  }

  return 0;
}

void ptx_thread_info::get_vector_operand_values(const operand_info &op,
                                                ptx_reg_t *ptx_regs,
                                                unsigned num_elements) {
  assert(op.is_vector());
  assert(num_elements <= 8);

  for (int idx = num_elements - 1; idx >= 0; --idx) {
    const symbol *sym = NULL;
    sym = op.vec_symbol(idx);
    if (strcmp(sym->name().c_str(), "_") != 0) {
      reg_map_t::iterator reg_iter = m_regs.back().find(sym);
      assert(reg_iter != m_regs.back().end());
      ptx_regs[idx] = reg_iter->second;
    }
  }
}

void sign_extend(ptx_reg_t &data, unsigned src_size, const operand_info &dst) {
  if (!dst.is_reg()) return;
  unsigned dst_size = get_operand_nbits(dst);
  if (src_size >= dst_size) return;
  // src_size < dst_size
  unsigned long long mask = 1;
  mask <<= (src_size - 1);
  if ((mask & data.u64) == 0) {
    // no need to sign extend
    return;
  }
  // need to sign extend
  mask = 1;
  mask <<= dst_size - src_size;
  mask -= 1;
  mask <<= src_size;
  data.u64 |= mask;
}

void ptx_thread_info::set_operand_value(const operand_info &dst,
                                        const ptx_reg_t &data, unsigned type,
                                        ptx_thread_info *thread,
                                        const ptx_instruction *pI, int overflow,
                                        int carry) {
  thread->set_operand_value(dst, data, type, thread, pI);

  if (dst.get_double_operand_type() == -2) {
    ptx_reg_t predValue;

    const symbol *sym = dst.vec_symbol(0);
    predValue.u64 = (m_regs.back()[sym].u64) & ~(0x0C);
    predValue.u64 |= ((overflow & 0x01) << 3);
    predValue.u64 |= ((carry & 0x01) << 2);

    set_reg(sym, predValue);
  } else if (dst.get_double_operand_type() == 0) {
    // intentionally do nothing
  } else {
    printf("Unexpected double destination\n");
    assert(0);
  }
}

void ptx_thread_info::set_operand_value(const operand_info &dst,
                                        const ptx_reg_t &data, unsigned type,
                                        ptx_thread_info *thread,
                                        const ptx_instruction *pI) {
  ptx_reg_t dstData;
  memory_space *mem = NULL;
  size_t size;
  int t;

  type_info_key::type_decode(type, size, t);

  /*complete this section for other cases*/
  if (dst.get_addr_space() == undefined_space) {
    ptx_reg_t setValue;
    setValue.u64 = data.u64;

    // Double destination in set instruction ($p0|$p1) - second is negation of
    // first
    if (dst.get_double_operand_type() == -1) {
      ptx_reg_t setValue2;
      const symbol *name1 = dst.vec_symbol(0);
      const symbol *name2 = dst.vec_symbol(1);

      if ((type == F16_TYPE) || (type == F32_TYPE) || (type == F64_TYPE) ||
          (type == FF64_TYPE)) {
        setValue2.f32 = (setValue.u64 == 0) ? 1.0f : 0.0f;
      } else {
        setValue2.u32 = (setValue.u64 == 0) ? 0xFFFFFFFF : 0;
      }

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
    }

    // Double destination in cvt,shr,mul,etc. instruction ($p0|$r4) - second
    // register operand receives data, first predicate operand is set as
    // $p0=($r4!=0) Also for Double destination in set instruction ($p0/$r1)
    else if ((dst.get_double_operand_type() == -2) ||
             (dst.get_double_operand_type() == -3)) {
      ptx_reg_t predValue;
      const symbol *predName = dst.vec_symbol(0);
      const symbol *regName = dst.vec_symbol(1);
      predValue.u64 = 0;

      switch (type) {
        case S8_TYPE:
          if ((setValue.s8 & 0x7F) == 0) predValue.u64 |= 1;
          break;
        case S16_TYPE:
          if ((setValue.s16 & 0x7FFF) == 0) predValue.u64 |= 1;
          break;
        case S32_TYPE:
          if ((setValue.s32 & 0x7FFFFFFF) == 0) predValue.u64 |= 1;
          break;
        case S64_TYPE:
          if ((setValue.s64 & 0x7FFFFFFFFFFFFFFF) == 0) predValue.u64 |= 1;
          break;
        case U8_TYPE:
        case B8_TYPE:
          if (setValue.u8 == 0) predValue.u64 |= 1;
          break;
        case U16_TYPE:
        case B16_TYPE:
          if (setValue.u16 == 0) predValue.u64 |= 1;
          break;
        case U32_TYPE:
        case B32_TYPE:
          if (setValue.u32 == 0) predValue.u64 |= 1;
          break;
        case U64_TYPE:
        case B64_TYPE:
          if (setValue.u64 == 0) predValue.u64 |= 1;
          break;
        case F16_TYPE:
          if (setValue.f16 == 0) predValue.u64 |= 1;
          break;
        case F32_TYPE:
          if (setValue.f32 == 0) predValue.u64 |= 1;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          if (setValue.f64 == 0) predValue.u64 |= 1;
          break;
        default:
          assert(0);
          break;
      }

      if ((type == S8_TYPE) || (type == S16_TYPE) || (type == S32_TYPE) ||
          (type == S64_TYPE) || (type == U8_TYPE) || (type == U16_TYPE) ||
          (type == U32_TYPE) || (type == U64_TYPE) || (type == B8_TYPE) ||
          (type == B16_TYPE) || (type == B32_TYPE) || (type == B64_TYPE)) {
        if ((setValue.u32 & (1 << (size - 1))) != 0) predValue.u64 |= 1 << 1;
      }
      if (type == F32_TYPE) {
        if (setValue.f32 < 0) predValue.u64 |= 1 << 1;
      }

      if (dst.get_operand_lohi() == 1) {
        setValue.u64 =
            ((m_regs.back()[regName].u64) & (~(0xFFFF))) + (data.u64 & 0xFFFF);
      } else if (dst.get_operand_lohi() == 2) {
        setValue.u64 = ((m_regs.back()[regName].u64) & (~(0xFFFF0000))) +
                       ((data.u64 << 16) & 0xFFFF0000);
      }

      set_reg(predName, predValue);
      set_reg(regName, setValue);
    } else if (type == BB128_TYPE) {
      // b128 stuff here.
      ptx_reg_t setValue2, setValue3, setValue4;
      setValue.u64 = 0;
      setValue2.u64 = 0;
      setValue3.u64 = 0;
      setValue4.u64 = 0;
      setValue.u32 = data.u128.lowest;
      setValue2.u32 = data.u128.low;
      setValue3.u32 = data.u128.high;
      setValue4.u32 = data.u128.highest;

      const symbol *name1, *name2, *name3, *name4 = NULL;

      name1 = dst.vec_symbol(0);
      name2 = dst.vec_symbol(1);
      name3 = dst.vec_symbol(2);
      name4 = dst.vec_symbol(3);

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
      set_reg(name3, setValue3);
      set_reg(name4, setValue4);
    } else if (type == BB64_TYPE || type == FF64_TYPE) {
      // ptxplus version of storing 64 bit values to registers stores to two
      // adjacent registers
      ptx_reg_t setValue2;
      setValue.u32 = 0;
      setValue2.u32 = 0;

      setValue.u32 = data.bits.ls;
      setValue2.u32 = data.bits.ms;

      const symbol *name1, *name2 = NULL;

      name1 = dst.vec_symbol(0);
      name2 = dst.vec_symbol(1);

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
    } else {
      if (dst.get_operand_lohi() == 1) {
        setValue.u64 = ((m_regs.back()[dst.get_symbol()].u64) & (~(0xFFFF))) +
                       (data.u64 & 0xFFFF);
      } else if (dst.get_operand_lohi() == 2) {
        setValue.u64 =
            ((m_regs.back()[dst.get_symbol()].u64) & (~(0xFFFF0000))) +
            ((data.u64 << 16) & 0xFFFF0000);
      }
      set_reg(dst.get_symbol(), setValue);
    }
  }

  // global memory - g[4], g[$r0]
  else if (dst.get_addr_space() == global_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = global_space;
  }

  // shared memory - s[4], s[$r0]
  else if (dst.get_addr_space() == shared_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->m_shared_mem;
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = shared_space;
  }

  // local memory - l0[4], l0[$r0]
  else if (dst.get_addr_space() == local_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->m_local_mem;
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = local_space;
  }

  else {
    printf("Destination stores to unknown location.");
    assert(0);
  }
}

void ptx_thread_info::set_vector_operand_values(const operand_info &dst,
                                                const ptx_reg_t &data1,
                                                const ptx_reg_t &data2,
                                                const ptx_reg_t &data3,
                                                const ptx_reg_t &data4) {
  unsigned num_elements = dst.get_vect_nelem();
  if (num_elements > 0) {
    set_reg(dst.vec_symbol(0), data1);
    if (num_elements > 1) {
      set_reg(dst.vec_symbol(1), data2);
      if (num_elements > 2) {
        set_reg(dst.vec_symbol(2), data3);
        if (num_elements > 3) {
          set_reg(dst.vec_symbol(3), data4);
        }
      }
    }
  }

  m_last_set_operand_value = data1;
}
void ptx_thread_info::set_wmma_vector_operand_values(
    const operand_info &dst, const ptx_reg_t &data1, const ptx_reg_t &data2,
    const ptx_reg_t &data3, const ptx_reg_t &data4, const ptx_reg_t &data5,
    const ptx_reg_t &data6, const ptx_reg_t &data7, const ptx_reg_t &data8) {
  unsigned num_elements = dst.get_vect_nelem();
  if (num_elements == 8) {
    set_reg(dst.vec_symbol(0), data1);
    set_reg(dst.vec_symbol(1), data2);
    set_reg(dst.vec_symbol(2), data3);
    set_reg(dst.vec_symbol(3), data4);
    set_reg(dst.vec_symbol(4), data5);
    set_reg(dst.vec_symbol(5), data6);
    set_reg(dst.vec_symbol(6), data7);
    set_reg(dst.vec_symbol(7), data8);
  } else {
    printf("error:set_wmma_vector_operands");
  }

  m_last_set_operand_value = data8;
}

#define my_abs(a) (((a) < 0) ? (-a) : (a))

#define MY_MAX_I(a, b) (a > b) ? a : b
#define MY_MAX_F(a, b) isNaN(a) ? b : isNaN(b) ? a : (a > b) ? a : b

#define MY_MIN_I(a, b) (a < b) ? a : b
#define MY_MIN_F(a, b) isNaN(a) ? b : isNaN(b) ? a : (a < b) ? a : b

#define MY_INC_I(a, b) (a >= b) ? 0 : a + 1
#define MY_DEC_I(a, b) ((a == 0) || (a > b)) ? b : a - 1

#define MY_CAS_I(a, b, c) (a == b) ? c : a

#define MY_EXCH(a, b) b

void abs_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case S16_TYPE:
      d.s16 = my_abs(a.s16);
      break;
    case S32_TYPE:
      d.s32 = my_abs(a.s32);
      break;
    case S64_TYPE:
      d.s64 = my_abs(a.s64);
      break;
    case U16_TYPE:
      d.s16 = my_abs(a.u16);
      break;
    case U32_TYPE:
      d.s32 = my_abs(a.u32);
      break;
    case U64_TYPE:
      d.s64 = my_abs(a.u64);
      break;
    case F32_TYPE:
      d.f32 = my_abs(a.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = my_abs(a.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void addp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // PTXPlus add instruction with carry (carry is kept in a predicate) register
  ptx_reg_t src1_data, src2_data, src3_data, data;
  int overflow = 0;
  int carry = 0;

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);

  unsigned rounding_mode = pI->rounding_mode();
  int orig_rm = fegetround();
  switch (rounding_mode) {
    case RN_OPTION:
      break;
    case RZ_OPTION:
      fesetround(FE_TOWARDZERO);
      break;
    default:
      assert(0);
      break;
  }

  // performs addition. Sets carry and overflow if needed.
  // src3_data.pred&0x4 is the carry flag
  switch (i_type) {
    case S8_TYPE:
      data.s64 = (src1_data.s64 & 0x0000000FF) + (src2_data.s64 & 0x0000000FF) +
                 (src3_data.pred & 0x4);
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) == 0) {
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000000100) >> 8;
      break;
    case S16_TYPE:
      data.s64 = (src1_data.s64 & 0x00000FFFF) + (src2_data.s64 & 0x00000FFFF) +
                 (src3_data.pred & 0x4);
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) == 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000010000) >> 16;
      break;
    case S32_TYPE:
      data.s64 = (src1_data.s64 & 0x0FFFFFFFF) + (src2_data.s64 & 0x0FFFFFFFF) +
                 (src3_data.pred & 0x4);
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) == 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 + src2_data.s64 + (src3_data.pred & 0x4);
      break;
    case U8_TYPE:
      data.u64 = (src1_data.u64 & 0xFF) + (src2_data.u64 & 0xFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x100) >> 8;
      break;
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) + (src2_data.u64 & 0xFFFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x10000) >> 16;
      break;
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) + (src2_data.u64 & 0xFFFFFFFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case U64_TYPE:
      data.s64 = src1_data.s64 + src2_data.s64 + (src3_data.pred & 0x4);
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 + src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 + src2_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 + src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }
  fesetround(orig_rm);

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);
}

void add_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  int overflow = 0;
  int carry = 0;

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  unsigned rounding_mode = pI->rounding_mode();
  int orig_rm = fegetround();
  switch (rounding_mode) {
    case RN_OPTION:
      break;
    case RZ_OPTION:
      fesetround(FE_TOWARDZERO);
      break;
    default:
      assert(0);
      break;
  }

  // performs addition. Sets carry and overflow if needed.
  switch (i_type) {
    case S8_TYPE:
      data.s64 = (src1_data.s64 & 0x0000000FF) + (src2_data.s64 & 0x0000000FF);
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) == 0) {
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000000100) >> 8;
      break;
    case S16_TYPE:
      data.s64 = (src1_data.s64 & 0x00000FFFF) + (src2_data.s64 & 0x00000FFFF);
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) == 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000010000) >> 16;
      break;
    case S32_TYPE:
      data.s64 = (src1_data.s64 & 0x0FFFFFFFF) + (src2_data.s64 & 0x0FFFFFFFF);
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) == 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 + src2_data.s64;
      break;
    case U8_TYPE:
      data.u64 = (src1_data.u64 & 0xFF) + (src2_data.u64 & 0xFF);
      carry = (data.u64 & 0x100) >> 8;
      break;
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) + (src2_data.u64 & 0xFFFF);
      carry = (data.u64 & 0x10000) >> 16;
      break;
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) + (src2_data.u64 & 0xFFFFFFFF);
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 + src2_data.u64;
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 + src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 + src2_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 + src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }
  fesetround(orig_rm);

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);
}

void addc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void and_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(~(src1_data.pred) & ~(src2_data.pred));
  else
    data.u64 = src1_data.u64 & src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void andn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case B16_TYPE:
      src2_data.u16 = ~src2_data.u16;
      break;
    case B32_TYPE:
      src2_data.u32 = ~src2_data.u32;
      break;
    case B64_TYPE:
      src2_data.u64 = ~src2_data.u64;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  data.u64 = src1_data.u64 & src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void bar_callback(const inst_t *inst, ptx_thread_info *thread) {
  unsigned ctaid = thread->get_cta_uid();
  unsigned barid = inst->bar_id;
  unsigned value = thread->get_reduction_value(ctaid, barid);
  const ptx_instruction *pI = dynamic_cast<const ptx_instruction *>(inst);
  const operand_info &dst = pI->dst();
  ptx_reg_t data;
  data.u32 = value;
  thread->set_operand_value(dst, value, U32_TYPE, thread, pI);
}

void atom_callback(const inst_t *inst, ptx_thread_info *thread) {
  const ptx_instruction *pI = dynamic_cast<const ptx_instruction *>(inst);

  // "Decode" the output type
  unsigned to_type = pI->get_type();
  size_t size;
  int t;
  type_info_key::type_decode(to_type, size, t);

  // Set up operand variables
  ptx_reg_t data;       // d
  ptx_reg_t src1_data;  // a
  ptx_reg_t src2_data;  // b
  ptx_reg_t op_result;  // temp variable to hold operation result

  bool data_ready = false;

  // Get operand info of sources and destination
  const operand_info &dst = pI->dst();    // d
  const operand_info &src1 = pI->src1();  // a
  const operand_info &src2 = pI->src2();  // b

  // Get operand values
  src1_data = thread->get_operand_value(src1, src1, to_type, thread, 1);  // a
  if (dst.get_symbol()->type()) {
    src2_data = thread->get_operand_value(src2, dst, to_type, thread, 1);  // b
  } else {
    // This is the case whent he first argument (dest) is '_'
    src2_data = thread->get_operand_value(src2, src1, to_type, thread, 1);  // b
  }

  // Check state space
  addr_t effective_address = src1_data.u64;
  memory_space_t space = pI->get_space();
  if (space == undefined_space) {
    // generic space - determine space via address
    if (whichspace(effective_address) == global_space) {
      effective_address = generic_to_global(effective_address);
      space = global_space;
    } else if (whichspace(effective_address) == shared_space) {
      unsigned smid = thread->get_hw_sid();
      effective_address = generic_to_shared(smid, effective_address);
      space = shared_space;
    } else {
      abort();
    }
  }
  assert(space == global_space || space == shared_space);

  memory_space *mem = NULL;
  if (space == global_space)
    mem = thread->get_global_memory();
  else if (space == shared_space)
    mem = thread->m_shared_mem;
  else
    abort();

  // Copy value pointed to in operand 'a' into register 'd'
  // (i.e. copy src1_data to dst)
  mem->read(effective_address, size / 8, &data.s64);
  if (dst.get_symbol()->type()) {
    thread->set_operand_value(dst, data, to_type, thread,
                              pI);  // Write value into register 'd'
  }

  // Get the atomic operation to be performed
  unsigned m_atomic_spec = pI->get_atomic();

  switch (m_atomic_spec) {
    // AND
    case ATOMIC_AND: {
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 & src2_data.u32;
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 & src2_data.s32;
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.AND "
              "only accepts b32\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // OR
    case ATOMIC_OR: {
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 | src2_data.u32;
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 | src2_data.s32;
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.OR "
              "only accepts b32\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // XOR
    case ATOMIC_XOR: {
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 ^ src2_data.u32;
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 ^ src2_data.s32;
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.XOR "
              "only accepts b32\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // CAS
    case ATOMIC_CAS: {
      ptx_reg_t src3_data;
      const operand_info &src3 = pI->src3();
      src3_data = thread->get_operand_value(src3, dst, to_type, thread, 1);

      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = MY_CAS_I(data.u32, src2_data.u32, src3_data.u32);
          data_ready = true;
          break;
        case B64_TYPE:
        case U64_TYPE:
          op_result.u64 = MY_CAS_I(data.u64, src2_data.u64, src3_data.u64);
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_CAS_I(data.s32, src2_data.s32, src3_data.s32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.CAS "
              "only accepts b32 and b64\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // EXCH
    case ATOMIC_EXCH: {
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = MY_EXCH(data.u32, src2_data.u32);
          data_ready = true;
          break;
        case B64_TYPE:
        case U64_TYPE:
          op_result.u64 = MY_EXCH(data.u64, src2_data.u64);
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_EXCH(data.s32, src2_data.s32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.EXCH "
              "only accepts b32\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // ADD
    case ATOMIC_ADD: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = data.u32 + src2_data.u32;
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 + src2_data.s32;
          data_ready = true;
          break;
        case U64_TYPE:
          op_result.u64 = data.u64 + src2_data.u64;
          data_ready = true;
          break;
        case F32_TYPE:
          op_result.f32 = data.f32 + src2_data.f32;
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.ADD only "
              "accepts u32, s32, u64, and f32\n");
          assert(0);
          break;
      }

      break;
    }
      // INC
    case ATOMIC_INC: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_INC_I(data.u32, src2_data.u32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.INC only "
              "accepts u32 and s32\n");
          assert(0);
          break;
      }

      break;
    }
      // DEC
    case ATOMIC_DEC: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_DEC_I(data.u32, src2_data.u32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.DEC only "
              "accepts u32 and s32\n");
          assert(0);
          break;
      }

      break;
    }
      // MIN
    case ATOMIC_MIN: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_MIN_I(data.u32, src2_data.u32);
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_MIN_I(data.s32, src2_data.s32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.MIN only "
              "accepts u32 and s32\n");
          assert(0);
          break;
      }

      break;
    }
      // MAX
    case ATOMIC_MAX: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_MAX_I(data.u32, src2_data.u32);
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_MAX_I(data.s32, src2_data.s32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.MAX only "
              "accepts u32 and s32\n");
          assert(0);
          break;
      }

      break;
    }
      // DEFAULT
    default: {
      assert(0);
      break;
    }
  }

  // Write operation result into  memory
  // (i.e. copy src1_data to dst)
  if (data_ready) {
    mem->write(effective_address, size / 8, &op_result.s64, thread, pI);
  } else {
    printf("Execution error: data_ready not set\n");
    assert(0);
  }
}

// atom_impl will now result in a callback being called in mem_ctrl_pop
// (gpu-sim.c)
void atom_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // SYNTAX
  // atom.space.operation.type d, a, b[, c]; (now read in callback)

  // obtain memory space of the operation
  memory_space_t space = pI->get_space();

  // get the memory address
  const operand_info &src1 = pI->src1();
  // const operand_info &dst  = pI->dst();  // not needed for effective address
  // calculation
  unsigned i_type = pI->get_type();
  ptx_reg_t src1_data;
  src1_data = thread->get_operand_value(src1, src1, i_type, thread, 1);
  addr_t effective_address = src1_data.u64;

  addr_t effective_address_final;

  // handle generic memory space by converting it to global
  if (space == undefined_space) {
    if (whichspace(effective_address) == global_space) {
      effective_address_final = generic_to_global(effective_address);
      space = global_space;
    } else if (whichspace(effective_address) == shared_space) {
      unsigned smid = thread->get_hw_sid();
      effective_address_final = generic_to_shared(smid, effective_address);
      space = shared_space;
    } else {
      abort();
    }
  } else {
    assert(space == global_space || space == shared_space);
    effective_address_final = effective_address;
  }

  // Check state space
  assert(space == global_space || space == shared_space);

  thread->m_last_effective_address = effective_address_final;
  thread->m_last_memory_space = space;
  thread->m_last_dram_callback.function = atom_callback;
  thread->m_last_dram_callback.instruction = pI;
}

void bar_impl(const ptx_instruction *pIin, ptx_thread_info *thread) {
  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);
  unsigned bar_op = pI->barrier_op();
  unsigned red_op = pI->get_atomic();
  unsigned ctaid = thread->get_cta_uid();

  switch (bar_op) {
    case SYNC_OPTION: {
      if (pI->get_num_operands() > 1) {
        const operand_info &op0 = pI->dst();
        const operand_info &op1 = pI->src1();
        ptx_reg_t op0_data;
        ptx_reg_t op1_data;
        op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1);
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);
        pI->set_bar_id(op0_data.u32);
        pI->set_bar_count(op1_data.u32);
      } else {
        const operand_info &op0 = pI->dst();
        ptx_reg_t op0_data;
        op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1);
        pI->set_bar_id(op0_data.u32);
      }
      break;
    }
    case ARRIVE_OPTION: {
      const operand_info &op0 = pI->dst();
      const operand_info &op1 = pI->src1();
      ptx_reg_t op0_data;
      ptx_reg_t op1_data;
      op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1);
      op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);
      pI->set_bar_id(op0_data.u32);
      pI->set_bar_count(op1_data.u32);
      break;
    }
    case RED_OPTION: {
      if (pI->get_num_operands() > 3) {
        const operand_info &op1 = pI->src1();
        const operand_info &op2 = pI->src2();
        const operand_info &op3 = pI->src3();
        ptx_reg_t op1_data;
        ptx_reg_t op2_data;
        ptx_reg_t op3_data;
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);
        op2_data = thread->get_operand_value(op2, op2, U32_TYPE, thread, 1);
        op3_data = thread->get_operand_value(op3, op3, PRED_TYPE, thread, 1);
        op3_data.u32 = !(op3_data.pred & 0x0001);
        pI->set_bar_id(op1_data.u32);
        pI->set_bar_count(op2_data.u32);
        switch (red_op) {
          case ATOMIC_POPC:
            thread->popc_reduction(ctaid, op1_data.u32, op3_data.u32);
            break;
          case ATOMIC_AND:
            thread->and_reduction(ctaid, op1_data.u32, op3_data.u32);
            break;
          case ATOMIC_OR:
            thread->or_reduction(ctaid, op1_data.u32, op3_data.u32);
            break;
          default:
            abort();
            break;
        }
      } else {
        const operand_info &op1 = pI->src1();
        const operand_info &op2 = pI->src2();
        ptx_reg_t op1_data;
        ptx_reg_t op2_data;
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);
        op2_data = thread->get_operand_value(op2, op2, PRED_TYPE, thread, 1);
        op2_data.u32 = !(op2_data.pred & 0x0001);
        pI->set_bar_id(op1_data.u32);
        pI->set_bar_count(thread->get_ntid().x * thread->get_ntid().y *
                          thread->get_ntid().z);
        switch (red_op) {
          case ATOMIC_POPC:
            thread->popc_reduction(ctaid, op1_data.u32, op2_data.u32);
            break;
          case ATOMIC_AND:
            thread->and_reduction(ctaid, op1_data.u32, op2_data.u32);
            break;
          case ATOMIC_OR:
            thread->or_reduction(ctaid, op1_data.u32, op2_data.u32);
            break;
          default:
            abort();
            break;
        }
      }
      break;
    }
    default:
      abort();
      break;
  }

  thread->m_last_dram_callback.function = bar_callback;
  thread->m_last_dram_callback.instruction = pIin;
}

void bfe_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  unsigned i_type = pI->get_type();
  unsigned msb = (i_type == U32_TYPE || i_type == S32_TYPE) ? 31 : 63;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  ptx_reg_t src = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1);
  ptx_reg_t data;
  unsigned pos = b.u32 & 0xFF;
  unsigned len = c.u32 & 0xFF;
  switch (i_type) {
    case U32_TYPE: {
      unsigned mask;
      data.u32 = src.u32 >> pos;
      mask = 0xFFFFFFFF >> (32 - len);
      data.u32 &= mask;
      break;
    }
    case U64_TYPE: {
      unsigned long mask;
      data.u64 = src.u64 >> pos;
      mask = 0xFFFFFFFFFFFFFFFF >> (64 - len);
      data.u64 &= mask;
      break;
    }
    case S32_TYPE: {
      unsigned mask;
      unsigned min = MY_MIN_I(pos + len - 1, msb);
      unsigned sbit = len == 0 ? 0 : (src.s32 >> min) & 0x1;
      data.s32 = src.s32 >> pos;
      if (sbit > 0) {
        mask = 0xFFFFFFFF << len;
        data.s32 |= mask;
      } else {
        mask = 0xFFFFFFFF >> (32 - len);
        data.s32 &= mask;
      }
      break;
    }
    case S64_TYPE: {
      unsigned long mask;
      unsigned min = MY_MIN_I(pos + len - 1, msb);
      unsigned sbit = len == 0 ? 0 : (src.s64 >> min) & 0x1;
      data.s64 = src.s64 >> pos;
      if (sbit > 0) {
        mask = 0xFFFFFFFFFFFFFFFF << len;
        data.s64 |= mask;
      } else {
        mask = 0xFFFFFFFFFFFFFFFF >> (64 - len);
        data.s64 &= mask;
      }
      break;
    }
    default:
      printf("Operand type not supported for BFE instruction.\n");
      abort();
      return;
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void bfi_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  int i, max;
  ptx_reg_t src1_data, src2_data;
  ptx_reg_t src3_data, src4_data, data;

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  const operand_info &src4 = pI->src4();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);
  src4_data = thread->get_operand_value(src4, dst, i_type, thread, 1);

  switch (i_type) {
    case B32_TYPE:
      max = 32;
      break;
    case B64_TYPE:
      max = 64;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }
  data = src2_data;
  unsigned pos = src3_data.u32 & 0xFF;
  unsigned len = src4_data.u32 & 0xFF;
  for (i = 0; i < len && pos + i < max; i++) {
    data.u32 = (~((0x00000001) << (pos + i))) & data.u32;
    data.u32 = data.u32 | ((src1_data.u32 & ((0x00000001) << (i))) << (pos));
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);
}
void bfind_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const unsigned i_type = pI->get_type();

  const ptx_reg_t src1_data =
      thread->get_operand_value(src1, dst, i_type, thread, 1);
  const int msb = (i_type == U32_TYPE || i_type == S32_TYPE) ? 31 : 63;

  unsigned long a = 0;
  switch (i_type) {
    case S32_TYPE:
      a = src1_data.s32;
      break;
    case U32_TYPE:
      a = src1_data.u32;
      break;
    case S64_TYPE:
      a = src1_data.s64;
      break;
    case U64_TYPE:
      a = src1_data.u64;
      break;
    default:
      assert(false);
      abort();
  }

  // negate negative signed inputs
  if ((i_type == S32_TYPE || i_type == S64_TYPE) && (a & (1 << msb))) {
    a = ~a;
  }
  uint32_t d_data = 0xffffffff;
  for (uint32_t i = msb; i >= 0; i--) {
    if (a & (1 << i)) {
      d_data = i;
      break;
    }
  }

  // if (.shiftamt && d != 0xffffffff)  { d = msb - d; }

  // store d
  thread->set_operand_value(dst, d_data, U32_TYPE, thread, pI);
}

void bra_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = pI->dst();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  thread->m_branch_taken = true;
  thread->set_npc(target_pc);
}

void brx_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = pI->dst();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  thread->m_branch_taken = true;
  thread->set_npc(target_pc);
}

void break_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = thread->pop_breakaddr();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  thread->m_branch_taken = true;
  thread->set_npc(target_pc);
}

void breakaddr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = pI->dst();
  thread->push_breakaddr(target);
  assert(
      pI->has_pred() ==
      false);  // pdom analysis cannot handle if this instruction is predicated
}

void brev_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);

  unsigned msb;
  switch (i_type) {
    case B32_TYPE:
      msb = 31;
      for (unsigned i = 0; i <= msb; i++) {
        if ((src1_data.u32 & (1 << i))) data.u32 |= 1 << (msb - i);
      }
      break;
    case B64_TYPE:
      msb = 63;
      for (unsigned i = 0; i <= msb; i++) {
        if ((src1_data.u64 & (1 << i))) data.u64 |= 1 << (msb - i);
      }
      break;
    default:
      assert(0);
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);
}
void brkpt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

unsigned trunc(unsigned num, unsigned precision) {
  int mask = 1, latest_one = -1;
  unsigned data = num;
  for (unsigned j = 0; j < sizeof(unsigned) * 8; j++) {
    int bit = data & mask;
    if (bit == 1) latest_one = j;
    data >>= 1;
  }
  if (latest_one >= precision) {
    // round_up is 1 if the most significant truncated digit is a 1, otherwise
    // it is 0
    // int round_up = (num & (1 << (latest_one-precision))) >>
    // (latest_one-precision); unsigned shifted_output = num >>
    // (latest_one-precision+1);
    // if shifted_output is a number like 1111, don't round up
    // if (shifted_output == (pow(2,precision)-1)) round_up = 0;
    // num = shifted_output + round_up;
    num >>= (latest_one - precision + 1);
  }
  return num;
}
void mapping(int thread, int wmma_type, int wmma_layout, int type, int index,
             int stride, int &row, int &col, int &assg_offset) {
  int offset;
  int c_row_offset[] = {0, 8, 0, 8, 4, 12, 4, 12};
  int c_col_offset[] = {0, 0, 8, 8, 0, 0, 8, 8};
  int c_tg_inside_row_offset[] = {0, 1, 0, 1};
  int c_tg_inside_col_offset[] = {0, 0, 2, 2};
  int c_inside_row_offset[] = {0, 0, 2, 2, 0, 0, 2, 2};
  int c_inside_col_offset[] = {0, 1, 0, 1, 4, 5, 4, 5};

  offset = thread_group_offset(thread, wmma_type, wmma_layout, type, stride);

  if (wmma_type == LOAD_A) {
    if (wmma_layout == ROW) {
      offset += index + 8 * ((thread % 16) / 8);
    } else {
      offset += 64 * (index / 4) + index % 4 + 128 * ((thread % 16) / 8);
    }
    offset = (offset / 16) * stride + offset % 16;
    assg_offset = index + 8 * ((thread % 16) / 8);
  } else if (wmma_type == LOAD_B) {
    if (wmma_layout == ROW) {
      offset += 64 * (index / 4) + index % 4 + 128 * ((thread % 16) / 8);
    } else {
      offset += index + 8 * ((thread % 16) / 8);
    }
    offset = (offset / 16) * stride + offset % 16;
    assg_offset = index + 8 * ((thread % 16) / 8);
  } else if (wmma_type == LOAD_C) {
    if (type == F16_TYPE) {
      row = c_row_offset[thread / 4] + thread % 4;
      col = c_col_offset[thread / 4] + index;
    } else {
      row = c_row_offset[thread / 4] + c_tg_inside_row_offset[thread % 4] +
            c_inside_row_offset[index];
      col = c_col_offset[thread / 4] + c_tg_inside_col_offset[thread % 4] +
            c_inside_col_offset[index];
    }
    assg_offset = index;
  }

  if (wmma_type == LOAD_A || wmma_type == LOAD_B) {
    if (wmma_layout == ROW) {
      row = offset / 16;
      col = offset % 16;
    } else {
      col = offset / 16;
      row = offset % 16;
    }
  }
}

void mma_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst) {
  int i, j, k, thrd;
  int row, col, offset;
  ptx_reg_t matrix_a[16][16];
  ptx_reg_t matrix_b[16][16];
  ptx_reg_t matrix_c[16][16];
  ptx_reg_t matrix_d[16][16];
  ptx_reg_t src_data;
  ptx_thread_info *thread;

  unsigned a_layout = pI->get_wmma_layout(0);
  unsigned b_layout = pI->get_wmma_layout(1);
  unsigned type = pI->get_type();
  unsigned type2 = pI->get_type2();
  
  // Check if this is an INT8 WMMA instruction (4 type parameters)
  // For INT8: m16n16k16.s32.s8.s8.s32, type2 should be s32 (accumulator)
  // For FP16: m16n16k16.f16.f16, type2 is the second type
  bool is_int8_wmma = (pI->get_scalar_type().size() == 4);
  
  int tid;
  const operand_info &dst = pI->operand_lookup(0);

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();
  float temp;
  half temp2;

  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    thread = core->get_thread_info()[tid + thrd];
    
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("THREAD=%d\n:", thrd);
    for (int operand_num = 1; operand_num <= 3; operand_num++) {
      const operand_info &src_a = pI->operand_lookup(operand_num);
      unsigned nelem = src_a.get_vect_nelem();
      
      // For INT8 WMMA: operand 1&2 have 2 elements, operand 3 has 8 elements
      // For FP16 WMMA: all operands have 4 or 8 elements
      if (is_int8_wmma && operand_num <= 2 && nelem != 2) {
        // Skip processing if INT8 WMMA input operands don't have expected 2 elements
        // This is a safety check - INT8 WMMA uses 2 registers for A/B inputs
        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
          printf("INT8 WMMA: operand %d has %d elements (expected 2)\n", operand_num, nelem);
        continue;
      }
      
      ptx_reg_t v[8];
      thread->get_vector_operand_values(src_a, v, nelem);

      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("Thread%d_Iteration=%d (nelem=%d)\n:", thrd, operand_num, nelem);
        for (k = 0; k < nelem; k++) {
          printf("%llx ", v[k].u64);
        }
        printf("\n");
      }
      ptx_reg_t nw_v[16];
      int hex_val;

      // Unpack register values based on data type
      if (!((operand_num == 3) && (type2 == F32_TYPE || type2 == S32_TYPE))) {
        if (is_int8_wmma && operand_num <= 2) {
          // INT8 WMMA: Unpack 4 INT8 values from each 32-bit register
          // For A/B operands: 2 registers × 4 INT8 = 8 INT8 values total
          for (k = 0; k < nelem; k++) {
            // Each register contains 4 INT8 values in bytes 0-3
            nw_v[k * 4 + 0].s32 = (v[k].s64 & 0xff);
            nw_v[k * 4 + 1].s32 = ((v[k].s64 >> 8) & 0xff);
            nw_v[k * 4 + 2].s32 = ((v[k].s64 >> 16) & 0xff);
            nw_v[k * 4 + 3].s32 = ((v[k].s64 >> 24) & 0xff);
            
            // Sign extend if s8 (not u8)
            if (type == S8_TYPE) {
              if (nw_v[k * 4 + 0].s32 & 0x80) nw_v[k * 4 + 0].s32 |= 0xffffff00;
              if (nw_v[k * 4 + 1].s32 & 0x80) nw_v[k * 4 + 1].s32 |= 0xffffff00;
              if (nw_v[k * 4 + 2].s32 & 0x80) nw_v[k * 4 + 2].s32 |= 0xffffff00;
              if (nw_v[k * 4 + 3].s32 & 0x80) nw_v[k * 4 + 3].s32 |= 0xffffff00;
            }
          }
          if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
            printf("INT8 unpacked:");
            for (k = 0; k < nelem * 4; k++) {
              printf("%d ", nw_v[k].s32);
            }
            printf("\n");
          }
        } else {
          // FP16 WMMA: Unpack 2 FP16 values from each 32-bit register
          for (k = 0; k < 2 * nelem; k++) {
            if (k % 2 == 1)
              hex_val = (v[k / 2].s64 & 0xffff);
            else
              hex_val = ((v[k / 2].s64 & 0xffff0000) >> 16);
            nw_v[k].f16 = *(reinterpret_cast<half *>(&hex_val));
          }
        }
      }
      if (!((operand_num == 3) && (type2 == F32_TYPE || type2 == S32_TYPE))) {
        if (is_int8_wmma && operand_num <= 2) {
          // INT8: Print integer values
          for (k = 0; k < nelem * 4; k++) {
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("%d ", nw_v[k].s32);
          }
        } else {
          // FP16: Print float values
          for (k = 0; k < 2 * nelem; k++) {
            temp = nw_v[k].f16;
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("%.2f ", temp);
          }
        }
        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) printf("\n");
      } else {
        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
          for (k = 0; k < nelem; k++) {
            if (type2 == S32_TYPE)
              printf("%d ", v[k].s32);
            else
              printf("%.2f ", v[k].f32);
          }
          printf("\n");
        }
      }
      
      // Use actual nelem instead of hardcoded 8 for flexibility
      // For INT8 WMMA: operand 1&2 have nelem=2, operand 3 has nelem=8
      // For FP16 WMMA: operand 1&2 have nelem=4 or 8, operand 3 has nelem=4 or 8
      unsigned loop_count;
      if (operand_num == 3) {
        loop_count = nelem;  // Use actual nelem for accumulator
      } else {
        // For A/B operands: INT8 uses nelem*4 (2*4=8), FP16 uses nelem*2 (4*2=8 or 8*2=16)
        if (is_int8_wmma) {
          loop_count = nelem * 4;  // INT8: 2 regs * 4 elements = 8 total
        } else {
          loop_count = nelem * 2;  // FP16: 4 regs * 2 elements = 8 total (or 8*2=16)
        }
      }
      
      switch (operand_num) {
        case 1:  // operand 1
          for (k = 0; k < loop_count; k++) {
            // Use correct type for mapping: INT8 for INT8 WMMA, F16 for FP16 WMMA
            unsigned mapping_type = is_int8_wmma ? type : F16_TYPE;
            mapping(thrd, LOAD_A, a_layout, mapping_type, k, 16, row, col, offset);
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("A:thread=%d,row=%d,col=%d,offset=%d,k=%d,loop_count=%d\n", thrd, row, col,
                     offset, k, loop_count);
            if (row < 16 && col < 16 && offset < (is_int8_wmma ? nelem * 4 : 16))
              matrix_a[row][col] = nw_v[offset];
          }
          break;
        case 2:  // operand 2
          for (k = 0; k < loop_count; k++) {
            // Use correct type for mapping: INT8 for INT8 WMMA, F16 for FP16 WMMA
            unsigned mapping_type = is_int8_wmma ? type : F16_TYPE;
            mapping(thrd, LOAD_B, b_layout, mapping_type, k, 16, row, col, offset);
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("B:thread=%d,row=%d,col=%d,offset=%d,k=%d,loop_count=%d\n", thrd, row, col,
                     offset, k, loop_count);
            if (row < 16 && col < 16 && offset < (is_int8_wmma ? nelem * 4 : 16))
              matrix_b[row][col] = nw_v[offset];
          }
          break;
        case 3:  // operand 3
          for (k = 0; k < loop_count; k++) {
            mapping(thrd, LOAD_C, ROW, type2, k, 16, row, col, offset);
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("C:thread=%d,row=%d,col=%d,offset=%d,k=%d,loop_count=%d\n", thrd, row, col,
                     offset, k, loop_count);
            if (row < 16 && col < 16 && offset < nelem) {
              if (type2 != F16_TYPE) {
                matrix_c[row][col] = v[offset];
              } else {
                matrix_c[row][col] = nw_v[offset];
              }
            }
          }
          break;
        default:
          printf("Invalid Operand Index\n");
      }
    }
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) printf("\n");
  }
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("MATRIX_A\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        temp = matrix_a[i][j].f16;
        printf("%.2f ", temp);
      }
      printf("\n");
    }
    printf("MATRIX_B\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        temp = matrix_b[i][j].f16;
        printf("%.2f ", temp);
      }
      printf("\n");
    }
    printf("MATRIX_C\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        if (type2 == F16_TYPE) {
          temp = matrix_c[i][j].f16;
          printf("%.2f ", temp);
        } else
          printf("%.2f ", matrix_c[i][j].f32);
      }
      printf("\n");
    }
  }
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
      matrix_d[i][j].f16 = 0;
    }
  }

  // Perform matrix multiplication: D = A * B + C
  if (is_int8_wmma) {
    // INT8 WMMA: Integer matrix multiplication
    // A, B contain INT8 values (stored as s32)
    // C, D contain INT32 values (s32)
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        // Initialize D[i][j] with accumulator C[i][j]
        matrix_d[i][j].s32 = matrix_c[i][j].s32;
        // Compute dot product: D[i][j] += sum(A[i][k] * B[k][j])
        for (k = 0; k < 16; k++) {
          matrix_d[i][j].s32 += matrix_a[i][k].s32 * matrix_b[k][j].s32;
        }
      }
    }
  } else {
    // FP16 WMMA: Floating-point matrix multiplication
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        for (k = 0; k < 16; k++) {
          matrix_d[i][j].f16 =
              matrix_d[i][j].f16 + matrix_a[i][k].f16 * matrix_b[k][j].f16;
        }
        if ((type == F16_TYPE) && (type2 == F16_TYPE))
          matrix_d[i][j].f16 += matrix_c[i][j].f16;
        else if ((type == F32_TYPE) && (type2 == F16_TYPE)) {
          temp2 = matrix_d[i][j].f16 + matrix_c[i][j].f16;
          temp = temp2;
          matrix_d[i][j].f32 = temp;
        } else if ((type == F16_TYPE) && (type2 == F32_TYPE)) {
          temp = matrix_d[i][j].f16;
          temp += matrix_c[i][j].f32;
          matrix_d[i][j].f16 = half(temp);
        } else {
          temp = matrix_d[i][j].f16;
          temp += matrix_c[i][j].f32;
          matrix_d[i][j].f32 = temp;
        }
      }
    }
  }
  
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("MATRIX_D\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        if (is_int8_wmma) {
          printf("%d ", matrix_d[i][j].s32);
        } else if (type == F16_TYPE) {
          temp = matrix_d[i][j].f16;
          printf("%.2f ", temp);
        } else
          printf("%.2f ", matrix_d[i][j].f32);
      }
      printf("\n");
    }
  }
  
  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    int row_t[8];
    int col_t[8];
    for (k = 0; k < 8; k++) {
      // Use type2 (output type) instead of type (input type) for mapping
      mapping(thrd, LOAD_C, ROW, type2, k, 16, row_t[k], col_t[k], offset);
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
        printf("mma:store:row:%d,col%d\n", row_t[k], col_t[k]);
    }
    thread = core->get_thread_info()[tid + thrd];

    if (is_int8_wmma) {
      // INT8 WMMA: Write back INT32 results
      thread->set_wmma_vector_operand_values(
          dst, matrix_d[row_t[0]][col_t[0]], matrix_d[row_t[1]][col_t[1]],
          matrix_d[row_t[2]][col_t[2]], matrix_d[row_t[3]][col_t[3]],
          matrix_d[row_t[4]][col_t[4]], matrix_d[row_t[5]][col_t[5]],
          matrix_d[row_t[6]][col_t[6]], matrix_d[row_t[7]][col_t[7]]);
      
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("thread%d:", thrd);
        for (k = 0; k < 8; k++) {
          printf("%d ", matrix_d[row_t[k]][col_t[k]].s32);
        }
        printf("\n");
      }
    } else if (type == F32_TYPE || type2 == F32_TYPE) {
      thread->set_wmma_vector_operand_values(
          dst, matrix_d[row_t[0]][col_t[0]], matrix_d[row_t[1]][col_t[1]],
          matrix_d[row_t[2]][col_t[2]], matrix_d[row_t[3]][col_t[3]],
          matrix_d[row_t[4]][col_t[4]], matrix_d[row_t[5]][col_t[5]],
          matrix_d[row_t[6]][col_t[6]], matrix_d[row_t[7]][col_t[7]]);

      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("thread%d:", thrd);
        for (k = 0; k < 8; k++) {
          printf("%.2f ", matrix_d[row_t[k]][col_t[k]].f32);
        }
        printf("\n");
      }
    } else if (type == F16_TYPE) {
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("thread%d:", thrd);
        for (k = 0; k < 8; k++) {
          temp = matrix_d[row_t[k]][col_t[k]].f16;
          printf("%.2f ", temp);
        }
        printf("\n");

        printf("thread%d:", thrd);
        for (k = 0; k < 8; k++) {
          printf("%x ", (unsigned int)matrix_d[row_t[k]][col_t[k]].f16);
        }
        printf("\n");
      }
      ptx_reg_t nw_data1, nw_data2, nw_data3, nw_data4;
      nw_data1.s64 = ((matrix_d[row_t[0]][col_t[0]].s64 & 0xffff)) |
                     ((matrix_d[row_t[1]][col_t[1]].s64 & 0xffff) << 16);
      nw_data2.s64 = ((matrix_d[row_t[2]][col_t[2]].s64 & 0xffff)) |
                     ((matrix_d[row_t[3]][col_t[3]].s64 & 0xffff) << 16);
      nw_data3.s64 = ((matrix_d[row_t[4]][col_t[4]].s64 & 0xffff)) |
                     ((matrix_d[row_t[5]][col_t[5]].s64 & 0xffff) << 16);
      nw_data4.s64 = ((matrix_d[row_t[6]][col_t[6]].s64 & 0xffff)) |
                     ((matrix_d[row_t[7]][col_t[7]].s64 & 0xffff) << 16);
      thread->set_vector_operand_values(dst, nw_data1, nw_data2, nw_data3,
                                        nw_data4);
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
        printf("thread%d=%llx,%llx,%llx,%llx", thrd, nw_data1.s64, nw_data2.s64,
               nw_data3.s64, nw_data4.s64);

    } else {
      printf("wmma:mma:wrong type\n");
      abort();
    }
  }
}

void call_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static unsigned call_uid_next = 1;

  const operand_info &target = pI->func_addr();
  assert(target.is_function_address());
  const symbol *func_addr = target.get_symbol();
  function_info *target_func = func_addr->get_pc();
  
  // Get function name early for FMR interception check
  std::string fname = target_func->get_name();
  printf("GPGPU-Sim PTX: Calling function: %s\n", fname.c_str());
  fflush(stdout); 
  
  // ========================================================================
  // FMR INTERCEPTION - Should never reach here (intercepted in ptx_exec_inst)
  // ========================================================================
  if (fname.find("fmr_sample") != std::string::npos) {
    printf("FATAL ERROR: FMR function %s should be intercepted before reaching this point.\n", fname.c_str());
    printf("FMR should be intercepted in ptx_exec_inst before reaching here.\n");
    printf("This indicates a bug in the FMR interception logic.\n");
    fflush(stdout);
    abort();
  }
  // ========================================================================
  // END FMR INTERCEPTION
  // ========================================================================
  
  if (target_func->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           target_func->get_name().c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           target_func->get_name().c_str());
    /*
     * Some of the instructions like printf() gives the gpgpusim the wrong
     * impression that it is a function call. As printf() doesnt have a body
     * like functions do, doing pdom analysis for printf() causes a crash.
     */
    if (target_func->get_function_size() > 0) target_func->do_pdom();
    target_func->set_pdom();
  }

  // check that number of args and return match function requirements
  if (pI->has_return() ^ target_func->has_return()) {
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
        "between\n"
        "               call instruction and function declaration\n");
    abort();
  }
  unsigned n_return = target_func->has_return();
  unsigned n_args = target_func->num_args();
  unsigned n_operands = pI->get_num_operands();

  if (n_operands != (n_return + 1 + n_args)) {
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of arguements "
        "between\n"
        "               call instruction and function declaration\n");
    abort();
  }

  // handle intrinsic functions
  if (fname == "vprintf") {
    gpgpusim_cuda_vprintf(pI, thread, target_func);
    return;
  }
#if (CUDART_VERSION >= 5000)
  // Jin: handle device runtime apis for CDP
  else if (fname == "cudaGetParameterBufferV2") {
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_getParameterBufferV2(
        pI, thread, target_func);
    return;
  } else if (fname == "cudaLaunchDeviceV2") {
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_launchDeviceV2(
        pI, thread, target_func);
    return;
  } else if (fname == "cudaStreamCreateWithFlags") {
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_streamCreateWithFlags(
        pI, thread, target_func);
    return;
  }
#endif

  // read source arguements into register specified in declaration of function
  arg_buffer_list_t arg_values;
  copy_args_into_buffer_list(pI, thread, target_func, arg_values);

  // record local for return value (we only support a single return value)
  const symbol *return_var_src = NULL;
  const symbol *return_var_dst = NULL;
  if (target_func->has_return()) {
    return_var_dst = pI->dst().get_symbol();
    return_var_src = target_func->get_return_var();
  }

  gpgpu_sim *gpu = thread->get_gpu();
  unsigned callee_pc = 0, callee_rpc = 0;
  if (gpu->simd_model() == POST_DOMINATOR) {
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc);
    assert(callee_pc == thread->get_pc());
  }

  thread->callstack_push(callee_pc + pI->inst_size(), callee_rpc,
                         return_var_src, return_var_dst, call_uid_next++);

  copy_buffer_list_into_frame(thread, arg_values);

  thread->set_npc(target_func);
}

// Ptxplus version of call instruction. Jumps to a label not a different Kernel.
void callp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static unsigned call_uid_next = 1;

  const operand_info &target = pI->dst();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  const symbol *return_var_src = NULL;
  const symbol *return_var_dst = NULL;

  gpgpu_sim *gpu = thread->get_gpu();
  unsigned callee_pc = 0, callee_rpc = 0;
  if (gpu->simd_model() == POST_DOMINATOR) {
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc);
    assert(callee_pc == thread->get_pc());
  }

  thread->callstack_push_plus(callee_pc + pI->inst_size(), callee_rpc,
                              return_var_src, return_var_dst, call_uid_next++);
  thread->set_npc(target_pc);
}

void clz_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  int max;
  unsigned long long mask;
  d.u64 = 0;

  switch (i_type) {
    case B32_TYPE:
      max = 32;
      mask = 0x80000000;
      break;
    case B64_TYPE:
      max = 64;
      mask = 0x8000000000000000;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  while ((d.u32 < max) && ((a.u64 & mask) == 0)) {
    d.u32++;
    a.u64 = a.u64 << 1;
  }

  thread->set_operand_value(dst, d, B32_TYPE, thread, pI);
}

void cnot_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case PRED_TYPE:
      d.pred = ((a.pred & 0x0001) == 0) ? 1 : 0;
      break;
    case B16_TYPE:
      d.u16 = (a.u16 == 0) ? 1 : 0;
      break;
    case B32_TYPE:
      d.u32 = (a.u32 == 0) ? 1 : 0;
      break;
    case B64_TYPE:
      d.u64 = (a.u64 == 0) ? 1 : 0;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void cos_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      d.f32 = cos(a.f32);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

ptx_reg_t chop(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  switch (to_width) {
    case 8:
      x.mask_and(0, 0xFF);
      break;
    case 16:
      x.mask_and(0, 0xFFFF);
      break;
    case 32:
      x.mask_and(0, 0xFFFFFFFF);
      break;
    case 64:
      break;
    default:
      assert(0);
  }
  return x;
}

ptx_reg_t sext(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  x = chop(x, 0, from_width, 0, rounding_mode, saturation_mode);
  switch (from_width) {
    case 8:
      if (x.get_bit(7)) x.mask_or(0xFFFFFFFF, 0xFFFFFF00);
      break;
    case 16:
      if (x.get_bit(15)) x.mask_or(0xFFFFFFFF, 0xFFFF0000);
      break;
    case 32:
      if (x.get_bit(31)) x.mask_or(0xFFFFFFFF, 0x00000000);
      break;
    case 64:
      break;
    default:
      assert(0);
  }
  return x;
}

// sign extend depending on the destination register size - hack to get
// SobelFilter working in CUDA 4.2
ptx_reg_t sexd(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  x = chop(x, 0, from_width, 0, rounding_mode, saturation_mode);
  switch (to_width) {
    case 8:
      if (x.get_bit(7)) x.mask_or(0xFFFFFFFF, 0xFFFFFF00);
      break;
    case 16:
      if (x.get_bit(15)) x.mask_or(0xFFFFFFFF, 0xFFFF0000);
      break;
    case 32:
      if (x.get_bit(31)) x.mask_or(0xFFFFFFFF, 0x00000000);
      break;
    case 64:
      break;
    default:
      assert(0);
  }
  return x;
}

ptx_reg_t zext(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  return chop(x, 0, from_width, 0, rounding_mode, saturation_mode);
}

int saturatei(int a, int max, int min) {
  if (a > max)
    a = max;
  else if (a < min)
    a = min;
  return a;
}

unsigned int saturatei(unsigned int a, unsigned int max) {
  if (a > max) a = max;
  return a;
}

ptx_reg_t f2x(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  half mytemp;
  half_float::half tmp_h;
  // assert( from_width == 32);

  enum cudaRoundMode mode = cudaRoundZero;
  switch (rounding_mode) {
    case RZI_OPTION:
      mode = cudaRoundZero;
      break;
    case RNI_OPTION:
      mode = cudaRoundNearest;
      break;
    case RMI_OPTION:
      mode = cudaRoundMinInf;
      break;
    case RPI_OPTION:
      mode = cudaRoundPosInf;
      break;
    default:
      break;
  }

  ptx_reg_t y;
  if (to_sign == 1) {  // convert to 64-bit number first?
    int tmp = cuda_math::float2int(x.f32, mode);
    if ((x.u32 & 0x7f800000) == 0) tmp = 0;  // round denorm. FP to 0
    if (saturation_mode && to_width < 32) {
      tmp = saturatei(tmp, (1 << to_width) - 1, -(1 << to_width));
    }
    switch (to_width) {
      case 8:
        y.s8 = (char)tmp;
        break;
      case 16:
        y.s16 = (short)tmp;
        break;
      case 32:
        y.s32 = (int)tmp;
        break;
      case 64:
        y.s64 = (long long)tmp;
        break;
      default:
        assert(0);
        break;
    }
  } else if (to_sign == 0) {
    unsigned int tmp = cuda_math::float2uint(x.f32, mode);
    if ((x.u32 & 0x7f800000) == 0) tmp = 0;  // round denorm. FP to 0
    if (saturation_mode && to_width < 32) {
      tmp = saturatei(tmp, (1 << to_width) - 1);
    }
    switch (to_width) {
      case 8:
        y.u8 = (unsigned char)tmp;
        break;
      case 16:
        y.u16 = (unsigned short)tmp;
        break;
      case 32:
        y.u32 = (unsigned int)tmp;
        break;
      case 64:
        y.u64 = (unsigned long long)tmp;
        break;
      default:
        assert(0);
        break;
    }
  } else {
    switch (to_width) {
      case 16:
        y.f16 = half_float::half_cast<half,
                                      std::numeric_limits<float>::round_style>(
            x.f32);  // mytemp;
        break;
      case 32:
        y.f32 = float(x.f16);
        break;  // handled by f2f
      case 64:
        y.f64 = x.f32;
        break;
      default:
        assert(0);
        break;
    }
  }
  return y;
}

double saturated2i(double a, double max, double min) {
  if (a > max)
    a = max;
  else if (a < min)
    a = min;
  return a;
}

ptx_reg_t d2x(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  assert(from_width == 64);

  double tmp;
  switch (rounding_mode) {
    case RZI_OPTION:
      tmp = trunc(x.f64);
      break;
    case RNI_OPTION:
      tmp = nearbyint(x.f64);
      break;
    case RMI_OPTION:
      tmp = floor(x.f64);
      break;
    case RPI_OPTION:
      tmp = ceil(x.f64);
      break;
    default:
      tmp = x.f64;
      break;
  }

  ptx_reg_t y;
  if (to_sign == 1) {
    tmp = saturated2i(tmp, ((1 << (to_width - 1)) - 1), (1 << (to_width - 1)));
    switch (to_width) {
      case 8:
        y.s8 = (char)tmp;
        break;
      case 16:
        y.s16 = (short)tmp;
        break;
      case 32:
        y.s32 = (int)tmp;
        break;
      case 64:
        y.s64 = (long long)tmp;
        break;
      default:
        assert(0);
        break;
    }
  } else if (to_sign == 0) {
    tmp = saturated2i(tmp, ((1 << (to_width - 1)) - 1), 0);
    switch (to_width) {
      case 8:
        y.u8 = (unsigned char)tmp;
        break;
      case 16:
        y.u16 = (unsigned short)tmp;
        break;
      case 32:
        y.u32 = (unsigned int)tmp;
        break;
      case 64:
        y.u64 = (unsigned long long)tmp;
        break;
      default:
        assert(0);
        break;
    }
  } else {
    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        y.f32 = x.f64;
        break;
      case 64:
        y.f64 = x.f64;  // should be handled by d2d
        break;
      default:
        assert(0);
        break;
    }
  }
  return y;
}

ptx_reg_t s2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y;

  if (from_width < 64) {  // 32-bit conversion
    y = sext(x, from_width, 32, 0, rounding_mode, saturation_mode);

    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        switch (rounding_mode) {
          case RZ_OPTION:
            y.f32 = cuda_math::__int2float_rz(y.s32);
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__int2float_rn(y.s32);
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__int2float_rd(y.s32);
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__int2float_ru(y.s32);
            break;
          default:
            break;
        }
        break;
      case 64:
        y.f64 = y.s32;
        break;  // no rounding needed
      default:
        assert(0);
        break;
    }
  } else {
    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        switch (rounding_mode) {
          case RZ_OPTION:
            y.f32 = cuda_math::__ll2float_rz(y.s64);
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__ll2float_rn(y.s64);
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__ll2float_rd(y.s64);
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__ll2float_ru(y.s64);
            break;
          default:
            break;
        }
        break;
      case 64:
        y.f64 = y.s64;
        break;  // no internal implementation found
      default:
        assert(0);
        break;
    }
  }

  // saturating an integer to 1 or 0?
  return y;
}

ptx_reg_t u2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y;

  if (from_width < 64) {  // 32-bit conversion
    y = zext(x, from_width, 32, 0, rounding_mode, saturation_mode);

    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        switch (rounding_mode) {
          case RZ_OPTION:
            y.f32 = cuda_math::__uint2float_rz(y.u32);
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__uint2float_rn(y.u32);
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__uint2float_rd(y.u32);
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__uint2float_ru(y.u32);
            break;
          default:
            break;
        }
        break;
      case 64:
        y.f64 = y.u32;
        break;  // no rounding needed
      default:
        assert(0);
        break;
    }
  } else {
    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        switch (rounding_mode) {
          case RZ_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64);
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64);
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64);
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64);
            break;
          default:
            break;
        }
        break;
      case 64:
        y.f64 = y.u64;
        break;  // no internal implementation found
      default:
        assert(0);
        break;
    }
  }

  // saturating an integer to 1 or 0?
  return y;
}

ptx_reg_t f2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y;
  if (from_width == 16) {
    half_float::detail::uint16 val = x.u16;
    y.f32 = half_float::detail::half2float<float>(val);
  } else {
    switch (rounding_mode) {
      case RZI_OPTION:
        y.f32 = truncf(x.f32);
        break;
      case RNI_OPTION:
#if CUDART_VERSION >= 3000
        y.f32 = nearbyintf(x.f32);
#else
        y.f32 = cuda_math::__internal_nearbyintf(x.f32);
#endif
        break;
      case RMI_OPTION:
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
        } else {
          y.f32 = floorf(x.f32);
        }
        break;
      case RPI_OPTION:
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
        } else {
          y.f32 = ceilf(x.f32);
        }
        break;
      default:
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
        } else {
          y.f32 = x.f32;
        }
        break;
    }
#if CUDART_VERSION >= 3000
    if (isnanf(y.f32))
#else
    if (cuda_math::__cuda___isnanf(y.f32))
#endif
    {
      y.u32 = 0x7fffffff;
    } else if (saturation_mode) {
      y.f32 = cuda_math::__saturatef(y.f32);
    }
  }

  return y;
}

ptx_reg_t d2d(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y;
  switch (rounding_mode) {
    case RZI_OPTION:
      y.f64 = trunc(x.f64);
      break;
    case RNI_OPTION:
#if CUDART_VERSION >= 3000
      y.f64 = nearbyint(x.f64);
#else
      y.f64 = cuda_math::__internal_nearbyintf(x.f64);
#endif
      break;
    case RMI_OPTION:
      y.f64 = floor(x.f64);
      break;
    case RPI_OPTION:
      y.f64 = ceil(x.f64);
      break;
    default:
      y.f64 = x.f64;
      break;
  }
  if (std::isnan(y.f64)) {
    y.u64 = 0xfff8000000000000ull;
  } else if (saturation_mode) {
    y.f64 = cuda_math::__saturatef(y.f64);
  }
  return y;
}

ptx_reg_t (*g_cvt_fn[11][11])(ptx_reg_t x, unsigned from_width,
                              unsigned to_width, int to_sign, int rounding_mode,
                              int saturation_mode) = {
    {NULL, sext, sext, sext, NULL, sext, sext, sext, s2f, s2f, s2f},
    {chop, NULL, sext, sext, chop, NULL, sext, sext, s2f, s2f, s2f},
    {chop, sexd, NULL, sext, chop, chop, NULL, sext, s2f, s2f, s2f},
    {chop, chop, chop, NULL, chop, chop, chop, NULL, s2f, s2f, s2f},
    {NULL, zext, zext, zext, NULL, zext, zext, zext, u2f, u2f, u2f},
    {chop, NULL, zext, zext, chop, NULL, zext, zext, u2f, u2f, u2f},
    {chop, chop, NULL, zext, chop, chop, NULL, zext, u2f, u2f, u2f},
    {chop, chop, chop, NULL, chop, chop, chop, NULL, u2f, u2f, u2f},
    {f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, NULL, f2f, f2x},
    {f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2f, f2x},
    {d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2d}};

void ptx_round(ptx_reg_t &data, int rounding_mode, int type) {
  if (rounding_mode == RN_OPTION) {
    return;
  }
  switch (rounding_mode) {
    case RZI_OPTION:
      switch (type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n");
          assert(0);
          break;
        case F16_TYPE:
          data.f16 = truncf(data.f16);
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = truncf(data.f32);
          break;
        case F64_TYPE:
        case FF64_TYPE:
          if (data.f64 < 0)
            data.f64 = ceil(data.f64);  // negative
          else
            data.f64 = floor(data.f64);  // positive
          break;
        default:
          assert(0);
          break;
      }
      break;
    case RNI_OPTION:
      switch (type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n");
          assert(0);
          break;
        case F16_TYPE:  // assert(0); break;
#if CUDART_VERSION >= 3000
          data.f16 = nearbyintf(data.f16);
#else
          data.f16 = cuda_math::__cuda_nearbyintf(data.f16);
#endif
          break;
        case F32_TYPE:
#if CUDART_VERSION >= 3000
          data.f32 = nearbyintf(data.f32);
#else
          data.f32 = cuda_math::__cuda_nearbyintf(data.f32);
#endif
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = round(data.f64);
          break;
        default:
          assert(0);
          break;
      }
      break;
    case RMI_OPTION:
      switch (type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n");
          assert(0);
          break;
        case F16_TYPE:
          data.f16 = floorf(data.f16);
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = floorf(data.f32);
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = floor(data.f64);
          break;
        default:
          assert(0);
          break;
      }
      break;
    case RPI_OPTION:
      switch (type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n");
          assert(0);
          break;
        case F16_TYPE:
          data.f16 = ceilf(data.f16);
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = ceilf(data.f32);
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = ceil(data.f64);
          break;
        default:
          assert(0);
          break;
      }
      break;
    default:
      break;
  }

  if (type == F32_TYPE) {
#if CUDART_VERSION >= 3000
    if (isnanf(data.f32))
#else
    if (cuda_math::__cuda___isnanf(data.f32))
#endif
    {
      data.u32 = 0x7fffffff;
    }
  }
  if ((type == F64_TYPE) || (type == FF64_TYPE)) {
    if (std::isnan(data.f64)) {
      data.u64 = 0xfff8000000000000ull;
    }
  }
}

void ptx_saturate(ptx_reg_t &data, int saturation_mode, int type) {
  if (!saturation_mode) {
    return;
  }
  switch (type) {
    case S8_TYPE:
    case S16_TYPE:
    case S32_TYPE:
    case S64_TYPE:
    case U8_TYPE:
    case U16_TYPE:
    case U32_TYPE:
    case U64_TYPE:
      printf("Trying to clamp an integer to 1??\n");
      assert(0);
      break;
    case F16_TYPE:                           // assert(0); break;
      if (data.f16 > 1.0f) data.f16 = 1.0f;  // negative
      if (data.f16 < 0.0f) data.f16 = 0.0f;  // positive
      break;
    case F32_TYPE:
      if (data.f32 > 1.0f) data.f32 = 1.0f;  // negative
      if (data.f32 < 0.0f) data.f32 = 0.0f;  // positive
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (data.f64 > 1.0f) data.f64 = 1.0f;  // negative
      if (data.f64 < 0.0f) data.f64 = 0.0f;  // positive
      break;
    default:
      assert(0);
      break;
  }
}

void cvt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  unsigned to_type = pI->get_type();
  unsigned from_type = pI->get_type2();
  unsigned rounding_mode = pI->rounding_mode();
  unsigned saturation_mode = pI->saturation_mode();

  //   if ( to_type == F16_TYPE || from_type == F16_TYPE )
  //      abort();

  int to_sign, from_sign;
  size_t from_width, to_width;
  unsigned src_fmt =
      type_info_key::type_decode(from_type, from_width, from_sign);
  unsigned dst_fmt = type_info_key::type_decode(to_type, to_width, to_sign);

  ptx_reg_t data = thread->get_operand_value(src1, dst, from_type, thread, 1);

  if (pI->is_neg()) {
    switch (from_type) {
      // Default to f32 for now, need to add support for others
      case S8_TYPE:
      case U8_TYPE:
      case B8_TYPE:
        data.s8 = -data.s8;
        break;
      case S16_TYPE:
      case U16_TYPE:
      case B16_TYPE:
        data.s16 = -data.s16;
        break;
      case S32_TYPE:
      case U32_TYPE:
      case B32_TYPE:
        data.s32 = -data.s32;
        break;
      case S64_TYPE:
      case U64_TYPE:
      case B64_TYPE:
        data.s64 = -data.s64;
        break;
      case F16_TYPE:
        data.f16 = -data.f16;
        break;
      case F32_TYPE:
        data.f32 = -data.f32;
        break;
      case F64_TYPE:
      case FF64_TYPE:
        data.f64 = -data.f64;
        break;
      default:
        assert(0);
    }
  }

  if (g_cvt_fn[src_fmt][dst_fmt] != NULL) {
    ptx_reg_t result = g_cvt_fn[src_fmt][dst_fmt](
        data, from_width, to_width, to_sign, rounding_mode, saturation_mode);
    data = result;
  }

  thread->set_operand_value(dst, data, to_type, thread, pI);
}

void cvta_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  memory_space_t space = pI->get_space();
  bool to_non_generic = pI->is_to();

  unsigned i_type = pI->get_type();
  ptx_reg_t from_addr = thread->get_operand_value(src1, dst, i_type, thread, 1);
  addr_t from_addr_hw = (addr_t)from_addr.u64;
  addr_t to_addr_hw = 0;
  unsigned smid = thread->get_hw_sid();
  unsigned hwtid = thread->get_hw_tid();

  if (to_non_generic) {
    switch (space.get_type()) {
      case shared_space:
        to_addr_hw = generic_to_shared(smid, from_addr_hw);
        break;
      case local_space:
        to_addr_hw = generic_to_local(smid, hwtid, from_addr_hw);
        break;
      case global_space:
        to_addr_hw = generic_to_global(from_addr_hw);
        break;
      default:
        abort();
    }
  } else {
    switch (space.get_type()) {
      case shared_space:
        to_addr_hw = shared_to_generic(smid, from_addr_hw);
        break;
      case local_space:
        to_addr_hw = local_to_generic(smid, hwtid, from_addr_hw) +
                     thread->get_local_mem_stack_pointer();
        break;  // add stack ptr here so that it can be passed as a pointer at
                // function call
      case global_space:
        to_addr_hw = global_to_generic(from_addr_hw);
        break;
      default:
        abort();
    }
  }

  ptx_reg_t to_addr;
  to_addr.u64 = to_addr_hw;
  thread->set_reg(dst.get_symbol(), to_addr);
}

void div_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();

  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case S8_TYPE:
      data.s8 = src1_data.s8 / src2_data.s8;
      break;
    case S16_TYPE:
      data.s16 = src1_data.s16 / src2_data.s16;
      break;
    case S32_TYPE:
      data.s32 = src1_data.s32 / src2_data.s32;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 / src2_data.s64;
      break;
    case U8_TYPE:
      data.u8 = src1_data.u8 / src2_data.u8;
      break;
    case U16_TYPE:
      data.u16 = src1_data.u16 / src2_data.u16;
      break;
    case U32_TYPE:
      data.u32 = src1_data.u32 / src2_data.u32;
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 / src2_data.u64;
      break;
    case B8_TYPE:
      data.u8 = src1_data.u8 / src2_data.u8;
      break;
    case B16_TYPE:
      data.u16 = src1_data.u16 / src2_data.u16;
      break;
    case B32_TYPE:
      data.u32 = src1_data.u32 / src2_data.u32;
      break;
    case B64_TYPE:
      data.u64 = src1_data.u64 / src2_data.u64;
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 / src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 / src2_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 / src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void dp4a_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  printf("DP4A instruction not implemented yet");
  assert(0);
}

void ex2_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();

  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      data.f32 = cuda_math::__powf(2.0, src1_data.f32);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void exit_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  thread->set_done();
  thread->exitCore();
  thread->registerExit();
}

void mad_def(const ptx_instruction *pI, ptx_thread_info *thread,
             bool use_carry = false);

void fma_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread);
}

void isspacep_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a;
  bool t = false;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  memory_space_t space = pI->get_space();

  a = thread->get_reg(src1.get_symbol());
  addr_t addr = (addr_t)a.u64;
  unsigned smid = thread->get_hw_sid();
  unsigned hwtid = thread->get_hw_tid();

  switch (space.get_type()) {
    case shared_space:
      t = isspace_shared(smid, addr);
    case local_space:
      t = isspace_local(smid, hwtid, addr);
    case global_space:
      t = isspace_global(addr);
    default:
      abort();
  }

  ptx_reg_t p;
  p.pred = t ? 1 : 0;

  thread->set_reg(dst.get_symbol(), p);
}

void decode_space(memory_space_t &space, ptx_thread_info *thread,
                  const operand_info &op, memory_space *&mem, addr_t &addr) {
  unsigned smid = thread->get_hw_sid();
  unsigned hwtid = thread->get_hw_tid();

  if (space == param_space_unclassified) {
    // need to op to determine whether it refers to a kernel param or local
    // param
    const symbol *s = op.get_symbol();
    const type_info *t = s->type();
    type_info_key ti = t->get_key();
    if (ti.is_param_kernel())
      space = param_space_kernel;
    else if (ti.is_param_local()) {
      space = param_space_local;
    }
    // mov r1, param-label
    else if (ti.is_reg()) {
      space = param_space_kernel;
    } else {
      printf("GPGPU-Sim PTX: ERROR ** cannot resolve .param space for '%s'\n",
             s->name().c_str());
      abort();
    }
  }
  switch (space.get_type()) {
    case global_space:
      mem = thread->get_global_memory();
      break;
    case param_space_local:
    case local_space:
      mem = thread->m_local_mem;
      addr += thread->get_local_mem_stack_pointer();
      break;
    case tex_space:
      mem = thread->get_tex_memory();
      break;
    case surf_space:
      mem = thread->get_surf_memory();
      break;
    case param_space_kernel:
      mem = thread->get_param_memory();
      break;
    case shared_space:
      mem = thread->m_shared_mem;
      break;
    case sstarr_space:
      mem = thread->m_sstarr_mem;
      break;
    case const_space:
      mem = thread->get_global_memory();
      break;
    case generic_space:
      if (thread->get_ptx_version().ver() >= 2.0) {
        // convert generic address to memory space address
        space = whichspace(addr);
        switch (space.get_type()) {
          case global_space:
            mem = thread->get_global_memory();
            addr = generic_to_global(addr);
            break;
          case local_space:
            mem = thread->m_local_mem;
            addr = generic_to_local(smid, hwtid, addr);
            break;
          case shared_space:
            mem = thread->m_shared_mem;
            addr = generic_to_shared(smid, addr);
            break;
          default:
            abort();
        }
      } else {
        abort();
      }
      break;
    case param_space_unclassified:
    case undefined_space:
    default:
      abort();
  }
}

void ld_exec(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned type = pI->get_type();

  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, type, thread, 1);
  ptx_reg_t data;
  memory_space_t space = pI->get_space();
  unsigned vector_spec = pI->get_vector();

  memory_space *mem = NULL;
  addr_t addr = src1_data.u32;

  decode_space(space, thread, src1, mem, addr);

  size_t size;
  int t;
  data.u64 = 0;
  type_info_key::type_decode(type, size, t);
  if (!vector_spec) {
    mem->read(addr, size / 8, &data.s64);
    if (type == S16_TYPE || type == S32_TYPE) sign_extend(data, size, dst);
    thread->set_operand_value(dst, data, type, thread, pI);
  } else {
    ptx_reg_t data1, data2, data3, data4;
    mem->read(addr, size / 8, &data1.s64);
    mem->read(addr + size / 8, size / 8, &data2.s64);
    if (vector_spec != V2_TYPE) {  // either V3 or V4
      mem->read(addr + 2 * size / 8, size / 8, &data3.s64);
      if (vector_spec != V3_TYPE) {  // v4
        mem->read(addr + 3 * size / 8, size / 8, &data4.s64);
        thread->set_vector_operand_values(dst, data1, data2, data3, data4);
      } else  // v3
        thread->set_vector_operand_values(dst, data1, data2, data3, data3);
    } else  // v2
      thread->set_vector_operand_values(dst, data1, data2, data2, data2);
  }
  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = space;
}

void ld_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ld_exec(pI, thread);
}

// ============================================================================
// FMR Implementation - Timing Model (Warp-Level)
// ============================================================================
void ld_sample_fmr_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  // FMR Tile Loader Timing Model - Warp-Cooperative Load with Interleaved SMEM
  // This function generates memory transactions for timing simulation AND performs functional simulation
  // Functional correctness is also handled here (no separate functional implementation needed)
  
  size_t elem_size = 2;  // f16 = 2 bytes
  unsigned smid;
  ptx_thread_info *thread;
  
  // Get operands (from intercepted CALL instruction)
  // For void return function: operand[n_return + 1 + arg]
  // n_return = 0 (void), so: operand[1 + arg]
  // operand[1]: arg0 - gmem_base_addr (u64)
  // operand[2]: arg1 - smem_base_addr (u64)
  // operand[3]: arg2 - width (s32)
  // operand[4]: arg3 - height (s32)
  // operand[5]: arg4 - stride (s32)
  const operand_info &dst = pI->dst();
  const operand_info &gmem_base_op = pI->operand_lookup(1);
  const operand_info &smem_base_op = pI->operand_lookup(2);
  const operand_info &width_op = pI->operand_lookup(3);
  const operand_info &height_op = pI->operand_lookup(4);
  const operand_info &stride_op = pI->operand_lookup(5);
  
  int tid = inst.warp_id() * core->get_warp_size();
  memory_space_t gmem_space = global_space;
  memory_space_t smem_space = shared_space;
  _memory_op_t insn_memory_op = memory_load;  // FMR is a load operation
  
  // Get tile parameters from first thread (uniform across warp)
  thread = core->get_thread_info()[tid];
  
  // Read parameters from local memory (CALL instruction passes params via .param space)
  assert(gmem_base_op.is_param_local());
  assert(smem_base_op.is_param_local());
  assert(width_op.is_param_local());
  assert(height_op.is_param_local());
  assert(stride_op.is_param_local());
  
  unsigned long long gmem_base;
  unsigned long long smem_base;
  int width, height, stride;
  
  thread->m_local_mem->read(gmem_base_op.get_symbol()->get_address(), sizeof(unsigned long long), &gmem_base);
  thread->m_local_mem->read(smem_base_op.get_symbol()->get_address(), sizeof(unsigned long long), &smem_base);
  thread->m_local_mem->read(width_op.get_symbol()->get_address(), sizeof(int), &width);
  thread->m_local_mem->read(height_op.get_symbol()->get_address(), sizeof(int), &height);
  thread->m_local_mem->read(stride_op.get_symbol()->get_address(), sizeof(int), &stride);
  
  smid = thread->get_hw_sid();
  
  // Parameter validation
  if (width <= 0 || height <= 0) {
    printf("FMR ERROR: Invalid tile dimensions: width=%d, height=%d\n", width, height);
    assert(0 && "FMR: Invalid tile dimensions");
  }
  if (stride < width) {
    printf("FMR WARNING: Stride (%d) < width (%d). This may cause incorrect memory access.\n", 
           stride, width);
  }
  
  // Address space conversion
  if (whichspace(gmem_base) == shared_space) {
    gmem_base = generic_to_shared(smid, gmem_base);
    gmem_space = shared_space;
  }
  smem_base = generic_to_shared(smid, smem_base);
  
  // Get memory space pointers for functional simulation
  memory_space *gmem = NULL;
  memory_space *smem = NULL;
  decode_space(gmem_space, thread, gmem_base_op, gmem, gmem_base);
  decode_space(smem_space, thread, smem_base_op, smem, smem_base);
  
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("FMR Tile Load: gmem=0x%llx, smem=0x%llx, w=%d, h=%d, stride=%d\n",
           (unsigned long long)gmem_base, (unsigned long long)smem_base, 
           width, height, stride);
  }
  
  // Store FMR tile metadata in warp_inst_t for statistics and debugging
  inst.m_fmr_tile_width = width;
  inst.m_fmr_tile_height = height;
  inst.m_fmr_stride = stride;
  inst.m_fmr_gmem_base = gmem_base;
  inst.m_fmr_smem_base = smem_base;
  
  // ========================================================================
  // TMA-like (Tensor Memory Accelerator-like) Load with Interleaved SMEM
  // Single-thread activation (thread 0) generates all addresses
  // Addresses are batched and stored in multiple thread slots
  // Similar to NVIDIA TMA address generation module
  // ========================================================================
  int warp_size = core->get_warp_size();
  
  // Step 1: Generate all GMEM addresses (TMA-like address generation)
  // Similar to hardware TMA, we generate all addresses upfront
  std::vector<new_addr_type> all_gmem_addrs;
  std::vector<std::pair<int, int> > addr_to_row_col;  // For functional simulation mapping
  
  for (int row = 0; row < height; row++) {
    int loads_per_row = (width + 7) / 8;  // Ceiling division: ceil(width / 8)
    for (int load = 0; load < loads_per_row; load++) {
      int start_col = load * 8;
      addr_t gmem_addr = gmem_base + (row * stride + start_col) * elem_size;
      all_gmem_addrs.push_back(gmem_addr);
      addr_to_row_col.push_back(std::make_pair(row, start_col));
    }
  }
  
  size_t total_txns = all_gmem_addrs.size();
  
  // Step 2: TMA-like address allocation - batch addresses into thread slots
  // Calculate number of batches needed
  int num_batches = (total_txns + MAX_ACCESSES_PER_INSN_PER_THREAD - 1) / 
                    MAX_ACCESSES_PER_INSN_PER_THREAD;
  
  // Limit to available thread slots
  if (num_batches > warp_size) {
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
      printf("FMR WARNING: Tile size (%dx%d) requires %d batches, "
             "but only %d thread slots available. Some transactions may be truncated.\n",
             width, height, num_batches, warp_size);
    }
    num_batches = warp_size;  // Limit to maximum available slots
  }
  
  // Clear all threads initially, then activate only slots that store addresses
  active_mask_t active_mask;
  active_mask.reset();
  
  // Batch addresses and store in thread slots
  for (int batch = 0; batch < num_batches; batch++) {
    size_t start_idx = batch * MAX_ACCESSES_PER_INSN_PER_THREAD;
    size_t end_idx = std::min(start_idx + MAX_ACCESSES_PER_INSN_PER_THREAD, total_txns);
    size_t batch_size = end_idx - start_idx;
    
    // Prepare batch address array
    new_addr_type batch_addrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    for (size_t i = 0; i < batch_size; i++) {
      batch_addrs[i] = all_gmem_addrs[start_idx + i];
    }
    // Fill remaining positions with 0 (indicates end of addresses)
    for (size_t i = batch_size; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++) {
      batch_addrs[i] = 0;
    }
    
    // Store addresses in thread slot 'batch'
    inst.set_addr(batch, batch_addrs, (unsigned)batch_size);
    
    // Activate this thread slot
    active_mask.set(batch);
  }
  
  // FIX: Do NOT modify inst.set_active() - keep all originally active threads active
  // Only batch slot threads will contribute memory accesses (via inst.set_addr()),
  // but all threads need to execute ptx_exec_inst() to advance their PC correctly.
  // Threads without addresses will have empty m_accessq, and generate_mem_accesses()
  // will naturally skip them.
  //
  // REMOVED: inst.set_active(active_mask);  // This caused PC desync

  // Step 3: Functional simulation - Thread 0 processes all data
  // This ensures data correctness regardless of batch limits
  thread = core->get_thread_info()[tid];  // Thread 0
  
  for (size_t addr_idx = 0; addr_idx < all_gmem_addrs.size(); addr_idx++) {
    int row = addr_to_row_col[addr_idx].first;
    int start_col = addr_to_row_col[addr_idx].second;
    int end_col = std::min(start_col + 8, width);
    
    // Process 8 elements (or fewer for last load in row)
    for (int col = start_col; col < end_col; col++) {
      addr_t gmem_read_addr = gmem_base + (row * stride + col) * elem_size;
      
      // Calculate SMEM address (tile-internal, continuous)
      int tile_idx = row * width + col;
      addr_t smem_write_addr = smem_base + tile_idx * elem_size;
      
      // Read from GMEM
      uint16_t data;
      gmem->read(gmem_read_addr, elem_size, &data);
      
      // Write to SMEM (interleaved bank layout)
      // NOTE: Bank interleaving is handled by SMEM hardware controller
      // We use continuous logical addressing; hardware maps to physical banks
      smem->write(smem_write_addr, elem_size, &data, thread, pI);
    }
  }
  
  // Step 4: Set memory transaction information for timing model
  // The memory system will use addresses stored in thread slots via generate_mem_accesses()
  inst.op = FMR_SAMPLE_OP;  // Set opcode so generate_mem_accesses() recognizes it
  inst.space = gmem_space;
  inst.data_size = 16;  // 128-bit = 16 bytes per transaction
  inst.memory_op = insn_memory_op;  // Explicitly set memory_op
  assert(inst.memory_op == insn_memory_op);
  
  // Step 5: Debug output
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("FMR Tile Load (TMA-like): gmem=0x%llx, smem=0x%llx, w=%d, h=%d, stride=%d\n",
           (unsigned long long)gmem_base, (unsigned long long)smem_base, 
           width, height, stride);
    printf("FMR: Total transactions=%d, Batches=%d, Active thread slots: ", 
           total_txns, num_batches);
    for (int i = 0; i < warp_size; i++) {
      if (active_mask.test(i)) printf("%d ", i);
    }
    printf("\n");
    printf("FMR: Interleaved Bank Layout - Row[even]→Banks[0,1], Row[odd]→Banks[2,3]\n");
    
    // Calculate bank distribution for first 4×4 elements (debug)
    int bank_counts[4] = {0, 0, 0, 0};
    for (int row = 0; row < std::min(height, 4); row++) {
      for (int col = 0; col < std::min(width, 4); col++) {
        int bank = ((row % 2) * 2) + (col % 2);
        bank_counts[bank]++;
      }
    }
    printf("FMR: Bank distribution (first 4×4): B0=%d, B1=%d, B2=%d, B3=%d\n",
           bank_counts[0], bank_counts[1], bank_counts[2], bank_counts[3]);
  }
}

// ============================================================================
// Deformable Attention Pipeline Implementation
// ============================================================================

// Stage 1: PCB (Pre-Check Block) - Weight Pruning
void deform_pcb_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  // PCB Stage: Pre-Check Block for weight pruning
  // CUDA signature: __deform_pcb(const float* weights, float threshold, bool* mask, int num_points, int level)

  // Early check: if functional simulation is disabled, just set latency and return
  gpgpu_sim *gpu = (gpgpu_sim*)core->get_gpu();
  inst.m_deform_stage = 1;  // PCB
  inst.latency = gpu->get_shader_config()->deform_pcb_latency;

  if (!gpu->get_shader_config()->gpgpu_deform_functional_sim_enabled) {
    if (gpu->gpgpu_ctx->debug_tensorcore) {
      printf("DEFORM_PCB: [BYPASS] functional_sim disabled, latency=%d\n", inst.latency);
    }
    return;
  }

  int tid = inst.warp_id() * core->get_warp_size();
  ptx_thread_info *thread = core->get_thread_info()[tid];

  int arg_base = 1;
  int n_args = 0;
  {
    const operand_info &target = pI->func_addr();
    if (target.is_function_address()) {
      function_info *target_func = target.get_symbol()->get_pc();
      arg_base = target_func->has_return() + 1;
      n_args = target_func->num_args();
    } else {
      arg_base = pI->has_return() ? 2 : 1;
      if (pI->get_num_operands() >= static_cast<unsigned>(arg_base)) {
        n_args = pI->get_num_operands() - arg_base;
      }
    }
  }
  if (n_args <= 0) {
    inst.latency = 1;
    return;
  }
  if (pI->get_num_operands() <= static_cast<unsigned>(arg_base)) {
    inst.latency = 1;
    return;
  }
  const operand_info &weights_op = pI->operand_lookup(arg_base + 0);
  const operand_info *threshold_op = NULL;
  const operand_info *mask_op = NULL;
  const operand_info *num_points_op = NULL;
  const operand_info *level_op = NULL;
  if (n_args == 4) {
    // Threshold optimized out: args = weights, mask, num_points, level
    mask_op = &pI->operand_lookup(arg_base + 1);
    num_points_op = &pI->operand_lookup(arg_base + 2);
    level_op = &pI->operand_lookup(arg_base + 3);
  } else {
    // Standard signature: weights, threshold, mask, num_points, level
    threshold_op = &pI->operand_lookup(arg_base + 1);
    mask_op = &pI->operand_lookup(arg_base + 2);
    num_points_op = &pI->operand_lookup(arg_base + 3);
    if (n_args > 4) {
      level_op = &pI->operand_lookup(arg_base + 4);
    }
  }
  
  // Read parameters from .param space (standard CUDA function call convention)
  unsigned long long weights_ptr, mask_ptr;
  float threshold;
  int num_points;
  int level = 0;
  
  thread->m_local_mem->read(weights_op.get_symbol()->get_address(),
                            sizeof(unsigned long long), &weights_ptr);
  if (threshold_op) {
    thread->m_local_mem->read(threshold_op->get_symbol()->get_address(),
                              sizeof(float), &threshold);
  } else {
    threshold = 0.0f;
  }
  if (mask_op) {
    thread->m_local_mem->read(mask_op->get_symbol()->get_address(),
                              sizeof(unsigned long long), &mask_ptr);
  }
  if (num_points_op) {
    thread->m_local_mem->read(num_points_op->get_symbol()->get_address(),
                              sizeof(int), &num_points);
  }
  if (level_op) {
    thread->m_local_mem->read(level_op->get_symbol()->get_address(),
                              sizeof(int), &level);
  }
  
  // Debug output (functional sim is enabled if we reach here)
  if (gpu->gpgpu_ctx->debug_tensorcore) {
    printf("DEFORM_PCB: weights_ptr=0x%llx, threshold=%.6f, mask_ptr=0x%llx, num_points=%d\n",
           (unsigned long long)weights_ptr, threshold, (unsigned long long)mask_ptr, num_points);
  }

  // Store metadata in instruction for timing model
  inst.m_deform_stage = 1;  // PCB

  // Block-level state for latency modeling
  kernel_info_t &k = thread->get_kernel();
  deform_block_state &blk_state = get_block_state(k, thread->get_ctaid(), level);

  // Address space conversion
  unsigned smid = thread->get_hw_sid();
  if (whichspace(weights_ptr) == shared_space) {
    weights_ptr = generic_to_shared(smid, weights_ptr);
  }
  if (whichspace(mask_ptr) == shared_space) {
    mask_ptr = generic_to_shared(smid, mask_ptr);
  }
  
  // Get memory spaces
  memory_space_t weights_space = global_space;
  memory_space_t mask_space = global_space;
  memory_space *weights_mem = NULL, *mask_mem = NULL;
  decode_space(weights_space, thread, weights_op, weights_mem, weights_ptr);
  decode_space(mask_space, thread, *mask_op, mask_mem, mask_ptr);
  
  // PCB Functional Simulation: Weight pruning
  int valid_count = 0;
  
  if (gpu->gpgpu_ctx->debug_tensorcore) {
    printf("DEFORM_PCB: Starting weight pruning (threshold=%.6f)\n", threshold);
  }
  
  for (int i = 0; i < num_points; i++) {
    float weight;
    weights_mem->read(weights_ptr + i * sizeof(float), sizeof(float), &weight);
    
    // Pruning logic: weight < threshold means invalid (low weight)
    bool valid = (fabsf(weight) >= threshold);
    uint8_t mask_bit = valid ? 0 : 1;  // 1 means invalid (should be skipped)
    
    mask_mem->write(mask_ptr + i, sizeof(uint8_t), &mask_bit, thread, pI);
    if (valid) valid_count++;
  }
  
  inst.m_deform_valid_points = valid_count;
  blk_state.num_points = num_points;
  blk_state.valid_points = valid_count;

  const shader_core_config *cfg = gpu->get_shader_config();
  if (!blk_state.pcb_latency_applied) {
    float scale = cfg->deform_global_scale *
                  level_scale_or_default(cfg, level) *
                  sparsity_factor_or_default(cfg, valid_count, num_points);
    float lat = static_cast<float>(cfg->deform_pcb_latency) * scale;
    inst.latency = clamp_latency(lat);
    blk_state.pcb_latency_applied = true;
  } else {
    inst.latency = 1;
  }
  
  if (gpu->gpgpu_ctx->debug_tensorcore) {
    printf("DEFORM_PCB: Pruning complete - valid_points=%d/%d (prune_rate=%.2f%%)\n",
           valid_count, num_points, 100.0f * (num_points - valid_count) / num_points);
  }
}

// Stage 2: GTC (Gated Tensor Core) - Operand Isolation
void deform_gtc_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  // GTC Stage: Gated Tensor Core for operand isolation
  // This stage runs in parallel with PCB (0 additional cycles)
  
  int tid = inst.warp_id() * core->get_warp_size();
  ptx_thread_info *thread = core->get_thread_info()[tid];
  
  // Minimal functional implementation - operand forwarding
  inst.m_deform_stage = 2;  // GTC
}

// Stage 3: TBC (Tile Boundary Check) - Aggregation Decision
void deform_tbc_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  // TBC Stage: Tile Boundary Check for aggregation decision
  // CUDA signature: int __deform_tbc(const float* coords, const bool* mask, int num_points,
  //                                   int* base_x, int* base_y, int* tile_w, int* tile_h, int level)

  // Early check: if functional simulation is disabled, just set latency and return
  gpgpu_sim *gpu = (gpgpu_sim*)core->get_gpu();
  inst.m_deform_stage = 3;  // TBC
  inst.m_deform_tile_mode = 0;  // Default: Horizontal mode
  inst.latency = gpu->get_shader_config()->deform_tbc_phase1_latency +
                 gpu->get_shader_config()->deform_tbc_phase2_latency;

  if (!gpu->get_shader_config()->gpgpu_deform_functional_sim_enabled) {
    if (gpu->gpgpu_ctx->debug_tensorcore) {
      printf("DEFORM_TBC: [BYPASS] functional_sim disabled, latency=%d\n", inst.latency);
    }
    return;
  }

  int tid = inst.warp_id() * core->get_warp_size();
  ptx_thread_info *thread = core->get_thread_info()[tid];

  // Store metadata
  inst.m_deform_stage = 3;  // TBC

  int arg_base = 1;
  {
    const operand_info &target = pI->func_addr();
    if (target.is_function_address()) {
      function_info *target_func = target.get_symbol()->get_pc();
      arg_base = target_func->has_return() + 1;
    } else {
      arg_base = pI->has_return() ? 2 : 1;
    }
  }
  if (pI->get_num_operands() <= static_cast<unsigned>(arg_base + 6)) {
    inst.latency = 1;
    return;
  }
  const operand_info &coords_op = pI->operand_lookup(arg_base + 0);
  const operand_info &mask_op = pI->operand_lookup(arg_base + 1);
  const operand_info &num_points_op = pI->operand_lookup(arg_base + 2);
  const operand_info &base_x_op = pI->operand_lookup(arg_base + 3);
  const operand_info &base_y_op = pI->operand_lookup(arg_base + 4);
  const operand_info &tile_w_op = pI->operand_lookup(arg_base + 5);
  const operand_info &tile_h_op = pI->operand_lookup(arg_base + 6);
  
  unsigned long long coords_ptr, mask_ptr, base_x_ptr, base_y_ptr, tile_w_ptr, tile_h_ptr;
  int num_points;
  int level = 0;
  
  thread->m_local_mem->read(coords_op.get_symbol()->get_address(), sizeof(unsigned long long), &coords_ptr);
  thread->m_local_mem->read(mask_op.get_symbol()->get_address(), sizeof(unsigned long long), &mask_ptr);
  thread->m_local_mem->read(num_points_op.get_symbol()->get_address(), sizeof(int), &num_points);
  thread->m_local_mem->read(base_x_op.get_symbol()->get_address(), sizeof(unsigned long long), &base_x_ptr);
  thread->m_local_mem->read(base_y_op.get_symbol()->get_address(), sizeof(unsigned long long), &base_y_ptr);
  thread->m_local_mem->read(tile_w_op.get_symbol()->get_address(), sizeof(unsigned long long), &tile_w_ptr);
  thread->m_local_mem->read(tile_h_op.get_symbol()->get_address(), sizeof(unsigned long long), &tile_h_ptr);
  if (pI->get_num_operands() > static_cast<unsigned>(arg_base + 7)) {
    const operand_info &level_op = pI->operand_lookup(arg_base + 7);
    thread->m_local_mem->read(level_op.get_symbol()->get_address(), sizeof(int), &level);
  }

  kernel_info_t &k = thread->get_kernel();
  deform_block_state &blk_state = get_block_state(k, thread->get_ctaid(), level);

  // Address space conversion (functional sim is enabled if we reach here)
  unsigned smid = thread->get_hw_sid();
  if (whichspace(coords_ptr) == shared_space) {
    coords_ptr = generic_to_shared(smid, coords_ptr);
  }
  if (whichspace(mask_ptr) == shared_space) {
    mask_ptr = generic_to_shared(smid, mask_ptr);
  }
  
  // Get memory spaces
  memory_space_t coords_space = global_space;
  memory_space_t mask_space = global_space;
  memory_space_t output_space = local_space;  // base_x/y/w/h are local variables
  memory_space *coords_mem = NULL, *mask_mem = NULL, *output_mem = NULL;
  decode_space(coords_space, thread, coords_op, coords_mem, coords_ptr);
  decode_space(mask_space, thread, mask_op, mask_mem, mask_ptr);
  output_mem = thread->m_local_mem;  // Output parameters are in local memory
  
  // TBC Functional Simulation: Compute bounding box of valid points
  float min_x = 1e9f, max_x = -1e9f;
  float min_y = 1e9f, max_y = -1e9f;
  int valid_count = 0;
  
  for (int i = 0; i < num_points; i++) {
    uint8_t mask_bit;
    mask_mem->read(mask_ptr + i, sizeof(uint8_t), &mask_bit);
    
    if (mask_bit == 0) {  // Valid point (mask=0 means not masked)
      float x, y;
      coords_mem->read(coords_ptr + i * 2 * sizeof(float), sizeof(float), &x);
      coords_mem->read(coords_ptr + (i * 2 + 1) * sizeof(float), sizeof(float), &y);
      
      min_x = fminf(min_x, x);
      max_x = fmaxf(max_x, x);
      min_y = fminf(min_y, y);
      max_y = fmaxf(max_y, y);
      valid_count++;
    }
  }
  
  // Compute tile parameters
  int base_x = (int)floorf(min_x);
  int base_y = (int)floorf(min_y);
  int tile_w = (int)ceilf(max_x) - base_x + 1;
  int tile_h = (int)ceilf(max_y) - base_y + 1;
  
  // Determine access mode
  int mode = 0;  // Default: Horizontal mode
  if (tile_w > 16 || tile_h > 16 || valid_count == 0) {
    mode = 3;  // DISCRETE mode (tile too large or no valid points)
  } else if (tile_h > tile_w) {
    mode = 1;  // VERTICAL mode
  } else if (tile_w == tile_h) {
    mode = 2;  // XOR mode
  }
  
  // Write output parameters
  output_mem->write(base_x_ptr, sizeof(int), &base_x, thread, pI);
  output_mem->write(base_y_ptr, sizeof(int), &base_y, thread, pI);
  output_mem->write(tile_w_ptr, sizeof(int), &tile_w, thread, pI);
  output_mem->write(tile_h_ptr, sizeof(int), &tile_h, thread, pI);
  
  // Store metadata
  inst.m_deform_tile_mode = mode;
  inst.m_deform_valid_points = valid_count;
  blk_state.num_points = num_points;
  blk_state.valid_points = valid_count;
  blk_state.mode = mode;
  blk_state.tile_w = tile_w;
  blk_state.tile_h = tile_h;

  const shader_core_config *cfg = gpu->get_shader_config();
  if (!blk_state.tbc_latency_applied) {
    float scale = cfg->deform_global_scale *
                  level_scale_or_default(cfg, level) *
                  sparsity_factor_or_default(cfg, valid_count, num_points);
    float base = static_cast<float>(cfg->deform_tbc_phase1_latency +
                                    cfg->deform_tbc_phase2_latency);
    int mode_idx = (mode < 0 || mode > 3) ? 0 : mode;
    float lat =
        base * scale + static_cast<float>(cfg->deform_mode_penalty[mode_idx]);
    inst.latency = clamp_latency(lat);
    blk_state.tbc_latency_applied = true;
  } else {
    inst.latency = 1;
  }
  
  if (gpu->gpgpu_ctx->debug_tensorcore) {
    const char* mode_names[] = {"HORIZONTAL", "VERTICAL", "XOR", "DISCRETE"};
    printf("DEFORM_TBC: bbox=[%.1f,%.1f]-[%.1f,%.1f], tile=%dx%d@(%d,%d), mode=%s, valid=%d\n",
           min_x, min_y, max_x, max_y, tile_w, tile_h, base_x, base_y, 
           mode_names[mode], valid_count);
  }
}

// Stage 4: TMA (Tile Memory Accelerator) - Memory Load
void deform_tma_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  // TMA Stage: Tile Memory Accelerator for feature map loading
  // CUDA signature: void __deform_tma(const float* src, float* dst, int tile_w, int tile_h,
  //                                    int src_pitch, int mode, int level)

  // Early check: if functional simulation is disabled, just set latency and return
  gpgpu_sim *gpu = (gpgpu_sim*)core->get_gpu();
  inst.m_deform_stage = 4;  // TMA
  inst.latency = gpu->get_shader_config()->deform_tma_latency +
                 gpu->get_shader_config()->deform_storage_latency;

  if (!gpu->get_shader_config()->gpgpu_deform_functional_sim_enabled) {
    if (gpu->gpgpu_ctx->debug_tensorcore) {
      printf("DEFORM_TMA: [BYPASS] functional_sim disabled, latency=%d\n", inst.latency);
    }
    return;
  }

  int tid = inst.warp_id() * core->get_warp_size();
  ptx_thread_info *thread = core->get_thread_info()[tid];

  int arg_base = 1;
  {
    const operand_info &target = pI->func_addr();
    if (target.is_function_address()) {
      function_info *target_func = target.get_symbol()->get_pc();
      arg_base = target_func->has_return() + 1;
    } else {
      arg_base = pI->has_return() ? 2 : 1;
    }
  }
  if (pI->get_num_operands() <= static_cast<unsigned>(arg_base + 5)) {
    inst.latency = 1;
    return;
  }
  const operand_info &src_op = pI->operand_lookup(arg_base + 0);
  const operand_info &dst_op = pI->operand_lookup(arg_base + 1);
  const operand_info &tile_w_op = pI->operand_lookup(arg_base + 2);
  const operand_info &tile_h_op = pI->operand_lookup(arg_base + 3);
  const operand_info &src_pitch_op = pI->operand_lookup(arg_base + 4);
  const operand_info &mode_op = pI->operand_lookup(arg_base + 5);
  
  unsigned long long src_ptr, dst_ptr;
  int tile_w, tile_h, src_pitch, mode;
  int level = 0;
  
  thread->m_local_mem->read(src_op.get_symbol()->get_address(), sizeof(unsigned long long), &src_ptr);
  thread->m_local_mem->read(dst_op.get_symbol()->get_address(), sizeof(unsigned long long), &dst_ptr);
  thread->m_local_mem->read(tile_w_op.get_symbol()->get_address(), sizeof(int), &tile_w);
  thread->m_local_mem->read(tile_h_op.get_symbol()->get_address(), sizeof(int), &tile_h);
  thread->m_local_mem->read(src_pitch_op.get_symbol()->get_address(), sizeof(int), &src_pitch);
  thread->m_local_mem->read(mode_op.get_symbol()->get_address(), sizeof(int), &mode);
  if (pI->get_num_operands() > static_cast<unsigned>(arg_base + 6)) {
    const operand_info &level_op = pI->operand_lookup(arg_base + 6);
    thread->m_local_mem->read(level_op.get_symbol()->get_address(), sizeof(int), &level);
  }

  kernel_info_t &k = thread->get_kernel();
  deform_block_state &blk_state = get_block_state(k, thread->get_ctaid(), level);

  // Store metadata
  inst.m_deform_stage = 4;  // TMA
  inst.m_deform_tile_mode = mode;

  // Debug output (functional sim is enabled if we reach here)
  if (gpu->gpgpu_ctx->debug_tensorcore) {
    printf("DEFORM_TMA: src=0x%llx, dst=0x%llx, tile=%dx%d, pitch=%d, mode=%d\n",
           (unsigned long long)src_ptr, (unsigned long long)dst_ptr,
           tile_w, tile_h, src_pitch, mode);
  }

  // Address space conversion
  unsigned smid = thread->get_hw_sid();
  if (whichspace(src_ptr) == global_space) {
    // src is in global memory (expected)
  } else if (whichspace(src_ptr) == shared_space) {
    src_ptr = generic_to_shared(smid, src_ptr);
  }
  
  if (whichspace(dst_ptr) == shared_space) {
    dst_ptr = generic_to_shared(smid, dst_ptr);
  }
  
  // Get memory spaces
  memory_space_t src_space = global_space;
  memory_space_t dst_space = shared_space;
  memory_space *src_mem = NULL, *dst_mem = NULL;
  decode_space(src_space, thread, src_op, src_mem, src_ptr);
  decode_space(dst_space, thread, dst_op, dst_mem, dst_ptr);
  
  // TMA Functional Simulation: Load tile_w × tile_h tile from global to shared memory
  // Destination uses pitch=17 to avoid bank conflicts (hardware feature)
  const int dst_pitch = 17;
  
  // Generate memory addresses for timing simulation
  std::vector<new_addr_type> mem_addrs;
  
  for (int row = 0; row < tile_h; row++) {
    for (int col = 0; col < tile_w; col++) {
      addr_t src_addr = src_ptr + (row * src_pitch + col) * sizeof(float);
      addr_t dst_addr = dst_ptr + (row * dst_pitch + col) * sizeof(float);
      mem_addrs.push_back(src_addr);
      
      // Functional copy
      float value;
      src_mem->read(src_addr, sizeof(float), &value);
      dst_mem->write(dst_addr, sizeof(float), &value, thread, pI);
    }
  }
  
  // Set up memory transactions for timing model
  int total_accesses = (int)mem_addrs.size();
  int batch_size = std::min(total_accesses, (int)MAX_ACCESSES_PER_INSN_PER_THREAD);
  new_addr_type batch_addrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
  
  for (int i = 0; i < batch_size; i++) {
    batch_addrs[i] = mem_addrs[i];
  }
  for (int i = batch_size; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++) {
    batch_addrs[i] = 0;
  }
  
  inst.set_addr(0, batch_addrs, batch_size);
  inst.space = global_space;
  inst.data_size = 4;  // 32-bit float
  inst.memory_op = memory_load;
  
  // Store metadata
  inst.m_deform_mem_accesses = total_accesses;
  blk_state.mode = mode;
  blk_state.tile_w = tile_w;
  blk_state.tile_h = tile_h;

  const shader_core_config *cfg = gpu->get_shader_config();
  if (!blk_state.tma_latency_applied) {
    float scale = cfg->deform_global_scale *
                  level_scale_or_default(cfg, level) *
                  sparsity_factor_or_default(cfg, blk_state.valid_points,
                                             blk_state.num_points);
    float base = static_cast<float>(cfg->deform_tma_latency +
                                    cfg->deform_storage_latency);
    float extra = cfg->deform_tma_per_elem *
                  static_cast<float>(tile_w * tile_h);
    float lat = base * scale + extra;
    inst.latency = clamp_latency(lat);
    blk_state.tma_latency_applied = true;
  } else {
    inst.latency = 1;
  }
  
  if (gpu->gpgpu_ctx->debug_tensorcore) {
    printf("DEFORM_TMA: Loaded %dx%d tile (%d accesses), mem_txns=%d\n",
           tile_w, tile_h, total_accesses, batch_size);
  }
}

// Stage 5: INTERP (Interpolation) - Bilinear Interpolation
void deform_interp_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  // INTERP Stage: Bilinear interpolation computation
  // CUDA signature: float __deform_interp(const float* tile_buffer, float local_x,
  //                                        float local_y, int mode, int level)

  // Early check: if functional simulation is disabled, just set latency and return
  gpgpu_sim *gpu = (gpgpu_sim*)core->get_gpu();
  inst.m_deform_stage = 5;  // INTERP
  inst.m_deform_interp_result = 0.0f;  // Dummy result
  inst.latency = gpu->get_shader_config()->deform_interp_latency;

  if (!gpu->get_shader_config()->gpgpu_deform_functional_sim_enabled) {
    if (gpu->gpgpu_ctx->debug_tensorcore) {
      printf("DEFORM_INTERP: [BYPASS] functional_sim disabled, latency=%d\n", inst.latency);
    }
    return;
  }

  int tid = inst.warp_id() * core->get_warp_size();
  ptx_thread_info *thread = core->get_thread_info()[tid];
  
  int arg_base = 1;
  {
    const operand_info &target = pI->func_addr();
    if (target.is_function_address()) {
      function_info *target_func = target.get_symbol()->get_pc();
      arg_base = target_func->has_return() + 1;
    } else {
      arg_base = pI->has_return() ? 2 : 1;
    }
  }
  if (pI->get_num_operands() <= static_cast<unsigned>(arg_base + 3)) {
    inst.latency = 1;
    return;
  }
  const operand_info &tile_buffer_op = pI->operand_lookup(arg_base + 0);
  const operand_info &local_x_op = pI->operand_lookup(arg_base + 1);
  const operand_info &local_y_op = pI->operand_lookup(arg_base + 2);
  const operand_info &mode_op = pI->operand_lookup(arg_base + 3);
  
  unsigned long long tile_buffer_ptr;
  float local_x, local_y;
  int mode;
  int level = 0;
  
  thread->m_local_mem->read(tile_buffer_op.get_symbol()->get_address(), sizeof(unsigned long long), &tile_buffer_ptr);
  thread->m_local_mem->read(local_x_op.get_symbol()->get_address(), sizeof(float), &local_x);
  thread->m_local_mem->read(local_y_op.get_symbol()->get_address(), sizeof(float), &local_y);
  thread->m_local_mem->read(mode_op.get_symbol()->get_address(), sizeof(int), &mode);
  if (pI->get_num_operands() > static_cast<unsigned>(arg_base + 4)) {
    const operand_info &level_op = pI->operand_lookup(arg_base + 4);
    thread->m_local_mem->read(level_op.get_symbol()->get_address(), sizeof(int), &level);
  }

  kernel_info_t &k = thread->get_kernel();
  deform_block_state &blk_state = get_block_state(k, thread->get_ctaid(), level);

  // Store metadata
  inst.m_deform_stage = 5;  // INTERP

  // Debug output (functional sim is enabled if we reach here)
  if (gpu->gpgpu_ctx->debug_tensorcore) {
    printf("DEFORM_INTERP: tile_buffer=0x%llx, local=(%.3f,%.3f), mode=%d\n",
           (unsigned long long)tile_buffer_ptr, local_x, local_y, mode);
  }

  // Address space conversion
  unsigned smid = thread->get_hw_sid();
  if (whichspace(tile_buffer_ptr) == shared_space) {
    tile_buffer_ptr = generic_to_shared(smid, tile_buffer_ptr);
  }
  
  // Get memory space
  memory_space_t tile_space = shared_space;
  memory_space *tile_mem = NULL;
  decode_space(tile_space, thread, tile_buffer_op, tile_mem, tile_buffer_ptr);
  
  // INTERP Functional Simulation: Bilinear interpolation
  // tile_buffer has pitch=17 to avoid bank conflicts
  const int tile_pitch = 17;
  
  // Compute interpolation weights
  int x0 = (int)floorf(local_x);
  int y0 = (int)floorf(local_y);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  
  float fx = local_x - x0;
  float fy = local_y - y0;
  
  // Read 4 corner values from tile buffer
  float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
  
  if (x0 >= 0 && y0 >= 0 && x0 < 16 && y0 < 16) {
    addr_t addr00 = tile_buffer_ptr + (y0 * tile_pitch + x0) * sizeof(float);
    tile_mem->read(addr00, sizeof(float), &v00);
  }
  if (x1 >= 0 && y0 >= 0 && x1 < 16 && y0 < 16) {
    addr_t addr01 = tile_buffer_ptr + (y0 * tile_pitch + x1) * sizeof(float);
    tile_mem->read(addr01, sizeof(float), &v01);
  }
  if (x0 >= 0 && y1 >= 0 && x0 < 16 && y1 < 16) {
    addr_t addr10 = tile_buffer_ptr + (y1 * tile_pitch + x0) * sizeof(float);
    tile_mem->read(addr10, sizeof(float), &v10);
  }
  if (x1 >= 0 && y1 >= 0 && x1 < 16 && y1 < 16) {
    addr_t addr11 = tile_buffer_ptr + (y1 * tile_pitch + x1) * sizeof(float);
    tile_mem->read(addr11, sizeof(float), &v11);
  }
  
  // Bilinear interpolation: result = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01 + (1-fx)*fy*v10 + fx*fy*v11
  float result = (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v01 + 
                 (1.0f - fx) * fy * v10 + fx * fy * v11;
  
  // Store metadata
  inst.m_deform_interp_result = result;  // Store for debugging

  const shader_core_config *cfg = gpu->get_shader_config();
  if (!blk_state.interp_latency_applied) {
    float scale = cfg->deform_global_scale *
                  level_scale_or_default(cfg, level) *
                  sparsity_factor_or_default(cfg, blk_state.valid_points,
                                             blk_state.num_points);
    float base = static_cast<float>(cfg->deform_interp_latency);
    float extra = cfg->deform_interp_per_point *
                  static_cast<float>(
                      blk_state.valid_points > 0 ? blk_state.valid_points
                                                 : blk_state.num_points);
    float lat = base * scale + extra;
    inst.latency = clamp_latency(lat);
    blk_state.interp_latency_applied = true;
  } else {
    inst.latency = 1;
  }
  
  if (gpu->gpgpu_ctx->debug_tensorcore) {
    printf("DEFORM_INTERP: corners=[%.3f,%.3f,%.3f,%.3f], weights=(%.3f,%.3f), result=%.6f\n",
           v00, v01, v10, v11, fx, fy, result);
  }
  
  // For functional simulation, we can skip storing the return value in register
  // as it will be used immediately in the kernel
}

void ldu_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ld_exec(pI, thread);
}

void mma_st_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  size_t size;
  unsigned smid;
  int t;
  int thrd, k;
  ptx_thread_info *thread;

  const operand_info &src = pI->operand_lookup(1);
  const operand_info &src1 = pI->operand_lookup(0);
  const operand_info &src2 = pI->operand_lookup(2);
  int tid;
  unsigned type = pI->get_type();
  unsigned wmma_type = pI->get_wmma_type();
  unsigned wmma_layout = pI->get_wmma_layout(0);
  int stride;

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  _memory_op_t insn_memory_op =
      pI->has_memory_read() ? memory_load : memory_store;
  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    thread = core->get_thread_info()[tid + thrd];
    ptx_reg_t addr_reg = thread->get_operand_value(src1, src, type, thread, 1);
    ptx_reg_t src2_data = thread->get_operand_value(src2, src, type, thread, 1);
    const operand_info &src_a = pI->operand_lookup(1);
    unsigned nelem = src_a.get_vect_nelem();
    ptx_reg_t *v = new ptx_reg_t[8];
    thread->get_vector_operand_values(src_a, v, nelem);
    stride = src2_data.u32;

    memory_space_t space = pI->get_space();

    memory_space *mem = NULL;
    addr_t addr = addr_reg.u32;

    new_addr_type mem_txn_addr[MAX_ACCESSES_PER_INSN_PER_THREAD];
    int num_mem_txn = 0;

    smid = thread->get_hw_sid();
    if (whichspace(addr) == shared_space) {
      addr = generic_to_shared(smid, addr);
      space = shared_space;
    }
    decode_space(space, thread, src1, mem, addr);

    type_info_key::type_decode(type, size, t);
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("mma_st: thrd=%d, addr=%x, fp(size=%zu), stride=%d\n", thrd,
             addr_reg.u32, size, src2_data.u32);
    addr_t new_addr =
        addr + thread_group_offset(thrd, wmma_type, wmma_layout, type, stride) *
                   size / 8;
    addr_t push_addr;

    ptx_reg_t nw_v[8];
    for (k = 0; k < 8; k++) {
      if (k % 2 == 0)
        nw_v[k].s64 = (v[k / 2].s64 & 0xffff);
      else
        nw_v[k].s64 = ((v[k / 2].s64 & 0xffff0000) >> 16);
    }

    for (k = 0; k < 8; k++) {
      if (type == F32_TYPE) {
        // mem->write(new_addr+4*acc_float_offset(k,wmma_layout,stride),size/8,&v[k].s64,thread,pI);
        push_addr = new_addr + 4 * acc_float_offset(k, wmma_layout, stride);
        mem->write(push_addr, size / 8, &v[k].s64, thread, pI);
        mem_txn_addr[num_mem_txn++] = push_addr;

        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
          printf(
              "wmma:store:thread%d=%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n",
              thrd, v[0].s64, v[1].s64, v[2].s64, v[3].s64, v[4].s64, v[5].s64,
              v[6].s64, v[7].s64);
          float temp;
          int l;
          printf("thread=%d:", thrd);
          for (l = 0; l < 8; l++) {
            temp = v[l].f32;
            printf("%.2f", temp);
          }
          printf("\n");
        }
      } else if (type == F16_TYPE) {
        if (wmma_layout == ROW) {
          // mem->write(new_addr+k*2,size/8,&nw_v[k].s64,thread,pI);
          push_addr = new_addr + k * 2;
          mem->write(push_addr, size / 8, &nw_v[k].s64, thread, pI);
          if (k % 2 == 0) mem_txn_addr[num_mem_txn++] = push_addr;
        } else if (wmma_layout == COL) {
          // mem->write(new_addr+k*2*stride,size/8,&nw_v[k].s64,thread,pI);
          push_addr = new_addr + k * 2 * stride;
          mem->write(push_addr, size / 8, &nw_v[k].s64, thread, pI);
          mem_txn_addr[num_mem_txn++] = push_addr;
        }

        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
          printf(
              "wmma:store:thread%d=%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n",
              thrd, nw_v[0].s64, nw_v[1].s64, nw_v[2].s64, nw_v[3].s64,
              nw_v[4].s64, nw_v[5].s64, nw_v[6].s64, nw_v[7].s64);
      }
    }

    delete[] v;
    inst.space = space;
    inst.set_addr(thrd, (new_addr_type *)mem_txn_addr, num_mem_txn);

    if ((type == F16_TYPE) &&
        (wmma_layout == COL))  // check the profiling xls for details
      inst.data_size = 2;      // 2 byte transaction
    else
      inst.data_size = 4;  // 4 byte transaction

    assert(inst.memory_op == insn_memory_op);
    // thread->m_last_effective_address = addr;
    // thread->m_last_memory_space = space;
  }
}

void mma_ld_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  size_t size;
  int t, i;
  unsigned smid;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned type = pI->get_type();
  unsigned wmma_type = pI->get_wmma_type();
  unsigned wmma_layout = pI->get_wmma_layout(0);
  
  int tid;
  int thrd, stride;
  ptx_thread_info *thread;

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  _memory_op_t insn_memory_op =
      pI->has_memory_read() ? memory_load : memory_store;

  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    thread = core->get_thread_info()[tid + thrd];
    ptx_reg_t src1_data =
        thread->get_operand_value(src1, dst, U32_TYPE, thread, 1);
    ptx_reg_t src2_data =
        thread->get_operand_value(src2, dst, U32_TYPE, thread, 1);
    stride = src2_data.u32;
    memory_space_t space = pI->get_space();

    memory_space *mem = NULL;
    addr_t addr = src1_data.u32;
    smid = thread->get_hw_sid();
    if (whichspace(addr) == shared_space) {
      addr = generic_to_shared(smid, addr);
      space = shared_space;
    }

    decode_space(space, thread, src1, mem, addr);
    type_info_key::type_decode(type, size, t);

    ptx_reg_t data[16];
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("mma_ld: thrd=%d,addr=%x, fpsize=%zu, stride=%d\n", thrd,
             src1_data.u32, size, src2_data.u32);

    addr_t new_addr =
        addr + thread_group_offset(thrd, wmma_type, wmma_layout, type, stride) *
                   size / 8;
    addr_t fetch_addr;
    new_addr_type mem_txn_addr[MAX_ACCESSES_PER_INSN_PER_THREAD];
    int num_mem_txn = 0;
    
    // Determine the number of elements to read based on destination register count
    // INT8 WMMA: dst has 2 registers, so we read 4 elements (2 per register)
    // FP16 WMMA: dst has 4 or 8 registers, so we read 8 or 16 elements
    unsigned dst_nelem = dst.get_vect_nelem();
    int num_elements_to_read = 16;  // default for FP16
    
    // Element stride in memory (bytes between consecutive elements)
    // For INT8: size=8 bits, so size/8=1 byte per element
    // For FP16: size=16 bits, so size/8=2 bytes per element
    int element_stride = (type == S8_TYPE || type == U8_TYPE) ? 1 : 2;
    
    if (wmma_type == LOAD_A || wmma_type == LOAD_B) {
      // For INT8: 2 registers (.v2) -> read 8 elements (4 INT8 per register)
      // For FP16: 8 half-registers (.v8) -> read 16 elements (2 FP16 per full register)
      if (dst_nelem == 2) {
        num_elements_to_read = 8;  // INT8 case: 8 INT8 values = 8 bytes = 2 × 32-bit regs
      } else {
        num_elements_to_read = 16;  // FP16 case
      }
    } else if (wmma_type == LOAD_C) {
      num_elements_to_read = 8;  // Both INT8 and FP16 use 8 for accumulator
    }

    if (wmma_type == LOAD_A) {
      for (i = 0; i < num_elements_to_read; i++) {
        if (wmma_layout == ROW) {
          // Row-major: consecutive elements in memory
          fetch_addr = new_addr + element_stride * i;
          mem->read(fetch_addr, size / 8, &data[i].s64);
        } else if (wmma_layout == COL) {
          // Column-major: elements separated by stride
          fetch_addr = new_addr + element_stride * (i % 4) + element_stride * stride * 4 * (i / 4);
          mem->read(fetch_addr, size / 8, &data[i].s64);
        } else {
          printf("mma_ld:wrong_layout_type\n");
          abort();
        }
        if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr;
      }
    } else if (wmma_type == LOAD_B) {
      for (i = 0; i < num_elements_to_read; i++) {
        if (wmma_layout == COL) {
          // Column-major: consecutive elements in memory
          fetch_addr = new_addr + element_stride * i;
          mem->read(fetch_addr, size / 8, &data[i].s64);
        } else if (wmma_layout == ROW) {
          // Row-major: elements separated by stride
          fetch_addr = new_addr + element_stride * (i % 4) + element_stride * stride * 4 * (i / 4);
          mem->read(fetch_addr, size / 8, &data[i].s64);
        } else {
          printf("mma_ld:wrong_layout_type\n");
          abort();
        }
        if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr;
      }
    } else if (wmma_type == LOAD_C) {
      for (i = 0; i < 8; i++) {
        if (type == F16_TYPE) {
          if (wmma_layout == ROW) {
            // mem->read(new_addr+2*i,size/8,&data[i].s64);
            fetch_addr = new_addr + 2 * i;
            mem->read(fetch_addr, size / 8, &data[i].s64);
            if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr;
          } else if (wmma_layout == COL) {
            // mem->read(new_addr+2*stride*i,size/8,&data[i].s64);
            fetch_addr = new_addr + 2 * stride * i;
            mem->read(fetch_addr, size / 8, &data[i].s64);
            mem_txn_addr[num_mem_txn++] = fetch_addr;
          } else {
            printf("mma_ld:wrong_type\n");
            abort();
          }
        } else if (type == F32_TYPE || type == S32_TYPE) {
          // F32 or S32 (INT8 WMMA uses S32 accumulator)
          // mem->read(new_addr+4*acc_float_offset(i,wmma_layout,stride),size/8,&data[i].s64);
          fetch_addr = new_addr + 4 * acc_float_offset(i, wmma_layout, stride);
          mem->read(fetch_addr, size / 8, &data[i].s64);
          mem_txn_addr[num_mem_txn++] = fetch_addr;
        } else {
          printf("mma_ld: unsupported type=%u for LOAD_C\n", type);
          abort();
        }
      }
    } else {
      printf("wrong wmma type\n");
      ;
      abort();
    }
    // generate timing memory request
    inst.space = space;
    inst.set_addr(thrd, (new_addr_type *)mem_txn_addr, num_mem_txn);

    if ((wmma_type == LOAD_C) && (type == F16_TYPE) &&
        (wmma_layout == COL))  // memory address is scattered, check the
                               // profiling xls for more detail.
      inst.data_size = 2;      // 2 byte transaction
    else
      inst.data_size = 4;  // 4 byte transaction
    assert(inst.memory_op == insn_memory_op);

    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
      if (type == F16_TYPE) {
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 16; i++) {
          printf("%llx ", data[i].u64);
        }
        printf("\n");

        printf("\nmma_ld:thread%d= ", thrd);
        float temp;
        for (i = 0; i < 16; i++) {
          temp = data[i].f16;
          printf("%.2f ", temp);
        }
        printf("\n");
      } else {
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 8; i++) {
          printf("%.2f ", data[i].f32);
        }
        printf("\n");
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 8; i++) {
          printf("%llx ", data[i].u64);
        }
        printf("\n");
      }
    }

    if ((wmma_type == LOAD_C) && (type == F32_TYPE || type == S32_TYPE)) {
      thread->set_wmma_vector_operand_values(dst, data[0], data[1], data[2],
                                             data[3], data[4], data[5], data[6],
                                             data[7]);
    } else {
      ptx_reg_t nw_data[8];
      int num_reg;
      
      // Determine number of registers based on actual destination operand size
      // INT8 WMMA (s8): LOAD_A/B use 2 registers, LOAD_C uses 8 registers
      // FP16 WMMA (f16): LOAD_A/B use 8 registers (4 packed), LOAD_C uses 4 registers
      unsigned dst_nelem = dst.get_vect_nelem();
      
      if (wmma_type == LOAD_C) {
        num_reg = (type == F16_TYPE) ? 4 : dst_nelem;
      } else {
        // For LOAD_A and LOAD_B
        num_reg = (dst_nelem == 2) ? 2 : 8;  // INT8 uses 2, FP16 uses 8
      }

      // Data packing: different for INT8 vs FP16
      bool is_int8 = (type == S8_TYPE || type == U8_TYPE);
      
      if (is_int8 && (wmma_type == LOAD_A || wmma_type == LOAD_B)) {
        // INT8 WMMA: Pack 4 INT8 values (1 byte each) into one 32-bit register
        // Each thread loads 8 INT8 values, pack into 2 registers
        // data[0..7] contain individual INT8 values (in lowest byte of each s64)
        
        // Initialize registers to zero first
        for (i = 0; i < 2; i++) {
          nw_data[i].u64 = 0;
        }
        
        // Pack data into registers
        for (i = 0; i < num_reg && i < 2; i++) {
          int start_idx = 4 * i;
          int end_idx = 4 * i + 3;
          
          if (end_idx >= num_elements_to_read) {
            break;  // Stop packing if we don't have enough data
          }
          
          // Pack 4 INT8 values into one 32-bit register
          nw_data[i].u64 = (data[start_idx].u64 & 0xff) |
                           ((data[start_idx + 1].u64 & 0xff) << 8) |
                           ((data[start_idx + 2].u64 & 0xff) << 16) |
                           ((data[start_idx + 3].u64 & 0xff) << 24);
        }
      } else {
        // FP16 WMMA: Pack 2 FP16 values (2 bytes each) into one 32-bit register
        for (i = 0; i < num_reg && i < 8; i++) {
          nw_data[i].s64 = ((data[2 * i].s64 & 0xffff) << 16) |
                           ((data[2 * i + 1].s64 & 0xffff));
        }
      }

      // Set registers based on actual number needed
      if (wmma_type == LOAD_C) {
        if (num_reg == 4)
          thread->set_vector_operand_values(dst, nw_data[0], nw_data[1],
                                            nw_data[2], nw_data[3]);
        else
          thread->set_wmma_vector_operand_values(
              dst, nw_data[0], nw_data[1], nw_data[2], nw_data[3], nw_data[4],
              nw_data[5], nw_data[6], nw_data[7]);
      } else {
        // LOAD_A or LOAD_B
        if (num_reg == 2) {
          // INT8 WMMA: only set 2 registers
          ptx_reg_t reg0, reg1, zero_data;
          reg0.u32 = nw_data[0];
          reg1.u32 = nw_data[1];
          zero_data.u64 = 0;
          
          thread->set_vector_operand_values(dst, reg0, reg1, zero_data, zero_data);
        } else {
          // FP16 WMMA: set 8 registers
          thread->set_wmma_vector_operand_values(
              dst, nw_data[0], nw_data[1], nw_data[2], nw_data[3], nw_data[4],
              nw_data[5], nw_data[6], nw_data[7]);
        }
      }
      
      // Print detailed register data only for FP16 WMMA to avoid accessing uninitialized memory in INT8 case
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore && !is_int8) {
        printf(
            "mma_ld:data[0].s64=%llx,data[1].s64=%llx,new_data[0].s64=%llx\n",
            data[0].u64, data[1].u64, nw_data[0].u64);
        printf(
            "mma_ld:data[2].s64=%llx,data[3].s64=%llx,new_data[1].s64=%llx\n",
            data[2].u64, data[3].u64, nw_data[1].u64);
        printf(
            "mma_ld:data[4].s64=%llx,data[5].s64=%llx,new_data[2].s64=%llx\n",
            data[4].u64, data[5].u64, nw_data[2].u64);
        printf(
            "mma_ld:data[6].s64=%llx,data[7].s64=%llx,new_data[3].s64=%llx\n",
            data[6].u64, data[7].u64, nw_data[3].u64);
        if (wmma_type != LOAD_C) {
          printf(
              "mma_ld:data[8].s64=%llx,data[9].s64=%llx,new_data[4].s64=%llx\n",
              data[8].u64, data[9].u64, nw_data[4].s64);
          printf(
              "mma_ld:data[10].s64=%llx,data[11].s64=%llx,new_data[5].s64=%"
              "llx\n",
              data[10].u64, data[11].u64, nw_data[5].u64);
          printf(
              "mma_ld:data[12].s64=%llx,data[13].s64=%llx,new_data[6].s64=%"
              "llx\n",
              data[12].u64, data[13].u64, nw_data[6].u64);
          printf(
              "mma_ld:data[14].s64=%llx,data[15].s64=%llx,new_data[7].s64=%"
              "llx\n",
              data[14].u64, data[15].u64, nw_data[3].u64);
        }
      }
    }

    // thread->m_last_effective_address = addr;
    // thread->m_last_memory_space = space;
  }
}

void lg2_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();

  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      d.f32 = log(a.f32) / log(2);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void mad24_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  ptx_reg_t d, t;

  unsigned i_type = pI->get_type();
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  unsigned sat_mode = pI->saturation_mode();

  assert(!pI->is_wide());

  switch (i_type) {
    case S32_TYPE:
      t.s64 = a.s32 * b.s32;
      if (pI->is_hi()) {
        d.s64 = (t.s64 >> 16) + c.s32;
        if (sat_mode) {
          if (d.s64 > (int)0x7FFFFFFF)
            d.s64 = (int)0x7FFFFFFF;
          else if (d.s64 < (int)0x80000000)
            d.s64 = (int)0x80000000;
        }
      } else if (pI->is_lo())
        d.s64 = t.s32 + c.s32;
      else
        assert(0);
      break;
    case U32_TYPE:
      t.u64 = a.u32 * b.u32;
      if (pI->is_hi())
        d.u64 = (t.u64 >> 16) + c.u32;
      else if (pI->is_lo())
        d.u64 = t.u32 + c.u32;
      else
        assert(0);
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void mad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, false);
}

void madp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, true);
}

void madc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, true);
}

void mad_def(const ptx_instruction *pI, ptx_thread_info *thread,
             bool use_carry) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  ptx_reg_t d, t;

  int carry = 0;
  int overflow = 0;

  unsigned i_type = pI->get_type();
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  // take the carry bit, it should be the 4th operand
  ptx_reg_t carry_bit;
  carry_bit.u64 = 0;
  if (use_carry) {
    const operand_info &carry = pI->operand_lookup(4);
    carry_bit = thread->get_operand_value(carry, dst, PRED_TYPE, thread, 0);
    carry_bit.pred &= 0x4;
    carry_bit.pred >>= 2;
  }

  unsigned rounding_mode = pI->rounding_mode();

  switch (i_type) {
    case S16_TYPE:
      t.s32 = a.s16 * b.s16;
      if (pI->is_wide())
        d.s32 = t.s32 + c.s32 + carry_bit.pred;
      else if (pI->is_hi())
        d.s16 = (t.s32 >> 16) + c.s16 + carry_bit.pred;
      else if (pI->is_lo())
        d.s16 = t.s16 + c.s16 + carry_bit.pred;
      else
        assert(0);
      carry =
          ((long long int)(t.s32 + c.s32 + carry_bit.pred) & 0x100000000) >> 32;
      break;
    case S32_TYPE:
      t.s64 = a.s32 * b.s32;
      if (pI->is_wide())
        d.s64 = t.s64 + c.s64 + carry_bit.pred;
      else if (pI->is_hi())
        d.s32 = (t.s64 >> 32) + c.s32 + carry_bit.pred;
      else if (pI->is_lo())
        d.s32 = t.s32 + c.s32 + carry_bit.pred;
      else
        assert(0);
      break;
    case S64_TYPE:
      t.s64 = a.s64 * b.s64;
      assert(!pI->is_wide());
      assert(!pI->is_hi());
      assert(use_carry == false);
      if (pI->is_lo())
        d.s64 = t.s64 + c.s64 + carry_bit.pred;
      else
        assert(0);
      break;
    case U16_TYPE:
      t.u32 = a.u16 * b.u16;
      if (pI->is_wide())
        d.u32 = t.u32 + c.u32 + carry_bit.pred;
      else if (pI->is_hi())
        d.u16 = (t.u32 + c.u16 + carry_bit.pred) >> 16;
      else if (pI->is_lo())
        d.u16 = t.u16 + c.u16 + carry_bit.pred;
      else
        assert(0);
      carry = ((long long int)((long long int)t.u32 + c.u32 + carry_bit.pred) &
               0x100000000) >>
              32;
      break;
    case U32_TYPE:
      t.u64 = a.u32 * b.u32;
      if (pI->is_wide())
        d.u64 = t.u64 + c.u64 + carry_bit.pred;
      else if (pI->is_hi())
        d.u32 = (t.u64 + c.u32 + carry_bit.pred) >> 32;
      else if (pI->is_lo())
        d.u32 = t.u32 + c.u32 + carry_bit.pred;
      else
        assert(0);
      break;
    case U64_TYPE:
      t.u64 = a.u64 * b.u64;
      assert(!pI->is_wide());
      assert(!pI->is_hi());
      assert(use_carry == false);
      if (pI->is_lo())
        d.u64 = t.u64 + c.u64 + carry_bit.pred;
      else
        assert(0);
      break;
    case F16_TYPE: {
      // assert(0);
      // break;
      assert(use_carry == false);
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }
      d.f16 = a.f16 * b.f16 + c.f16;
      if (pI->saturation_mode()) {
        if (d.f16 < 0)
          d.f16 = 0;
        else if (d.f16 > 1.0f)
          d.f16 = 1.0f;
      }
      fesetround(orig_rm);
      break;
    }
    case F32_TYPE: {
      assert(use_carry == false);
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          // assert(0);
          break;
      }
      d.f32 = a.f32 * b.f32 + c.f32;
      if (pI->saturation_mode()) {
        if (d.f32 < 0)
          d.f32 = 0;
        else if (d.f32 > 1.0f)
          d.f32 = 1.0f;
      }
      fesetround(orig_rm);
      break;
    }
    case F64_TYPE:
    case FF64_TYPE: {
      assert(use_carry == false);
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }
      d.f64 = a.f64 * b.f64 + c.f64;
      if (pI->saturation_mode()) {
        if (d.f64 < 0)
          d.f64 = 0;
        else if (d.f64 > 1.0f)
          d.f64 = 1.0;
      }
      fesetround(orig_rm);
      break;
    }
    default:
      assert(0);
      break;
  }
  thread->set_operand_value(dst, d, i_type, thread, pI, overflow, carry);
}

bool isNaN(float x) { return std::isnan(x); }

bool isNaN(double x) { return std::isnan(x); }

void max_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
      d.u16 = MY_MAX_I(a.u16, b.u16);
      break;
    case U32_TYPE:
      d.u32 = MY_MAX_I(a.u32, b.u32);
      break;
    case U64_TYPE:
      d.u64 = MY_MAX_I(a.u64, b.u64);
      break;
    case S16_TYPE:
      d.s16 = MY_MAX_I(a.s16, b.s16);
      break;
    case S32_TYPE:
      d.s32 = MY_MAX_I(a.s32, b.s32);
      break;
    case S64_TYPE:
      d.s64 = MY_MAX_I(a.s64, b.s64);
      break;
    case F32_TYPE:
      d.f32 = MY_MAX_F(a.f32, b.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = MY_MAX_F(a.f64, b.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void membar_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // handled by timing simulator
}

void min_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
      d.u16 = MY_MIN_I(a.u16, b.u16);
      break;
    case U32_TYPE:
      d.u32 = MY_MIN_I(a.u32, b.u32);
      break;
    case U64_TYPE:
      d.u64 = MY_MIN_I(a.u64, b.u64);
      break;
    case S16_TYPE:
      d.s16 = MY_MIN_I(a.s16, b.s16);
      break;
    case S32_TYPE:
      d.s32 = MY_MIN_I(a.s32, b.s32);
      break;
    case S64_TYPE:
      d.s64 = MY_MIN_I(a.s64, b.s64);
      break;
    case F32_TYPE:
      d.f32 = MY_MIN_F(a.f32, b.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = MY_MIN_F(a.f64, b.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void mov_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  unsigned i_type = pI->get_type();
  assert(src1.is_param_local() == 0);

  if ((src1.is_vector() || dst.is_vector()) && (i_type != BB64_TYPE) &&
      (i_type != BB128_TYPE) && (i_type != FF64_TYPE)) {
    // pack or unpack operation
    unsigned nbits_to_move;
    ptx_reg_t tmp_bits;

    switch (pI->get_type()) {
      case B16_TYPE:
        nbits_to_move = 16;
        break;
      case B32_TYPE:
        nbits_to_move = 32;
        break;
      case B64_TYPE:
        nbits_to_move = 64;
        break;
      default:
        printf(
            "Execution error: mov pack/unpack with unsupported type "
            "qualifier\n");
        assert(0);
        break;
    }

    if (src1.is_vector()) {
      unsigned nelem = src1.get_vect_nelem();
      ptx_reg_t v[4];
      thread->get_vector_operand_values(src1, v, nelem);

      unsigned bits_per_src_elem = nbits_to_move / nelem;
      for (unsigned i = 0; i < nelem; i++) {
        switch (bits_per_src_elem) {
          case 8:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u8) << (8 * i));
            break;
          case 16:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u16) << (16 * i));
            break;
          case 32:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u32) << (32 * i));
            break;
          default:
            printf(
                "Execution error: mov pack/unpack with unsupported source/dst "
                "size ratio (src)\n");
            assert(0);
            break;
        }
      }
    } else {
      data = thread->get_operand_value(src1, dst, i_type, thread, 1);

      switch (pI->get_type()) {
        case B16_TYPE:
          tmp_bits.u16 = data.u16;
          break;
        case B32_TYPE:
          tmp_bits.u32 = data.u32;
          break;
        case B64_TYPE:
          tmp_bits.u64 = data.u64;
          break;
        default:
          assert(0);
          break;
      }
    }

    if (dst.is_vector()) {
      unsigned nelem = dst.get_vect_nelem();
      ptx_reg_t v[4];
      unsigned bits_per_dst_elem = nbits_to_move / nelem;
      for (unsigned i = 0; i < nelem; i++) {
        switch (bits_per_dst_elem) {
          case 8:
            v[i].u8 = (tmp_bits.u64 >> (8 * i)) & ((unsigned long long)0xFF);
            break;
          case 16:
            v[i].u16 =
                (tmp_bits.u64 >> (16 * i)) & ((unsigned long long)0xFFFF);
            break;
          case 32:
            v[i].u32 =
                (tmp_bits.u64 >> (32 * i)) & ((unsigned long long)0xFFFFFFFF);
            break;
          default:
            printf(
                "Execution error: mov pack/unpack with unsupported source/dst "
                "size ratio (dst)\n");
            assert(0);
            break;
        }
      }
      thread->set_vector_operand_values(dst, v[0], v[1], v[2], v[3]);
    } else {
      thread->set_operand_value(dst, tmp_bits, i_type, thread, pI);
    }
  } else if (i_type == PRED_TYPE and src1.is_literal() == true) {
    // in ptx, literal input translate to predicate as 0 = false and 1 = true
    // we have adopted the opposite to simplify implementation of zero flags in
    // ptxplus
    data = thread->get_operand_value(src1, dst, i_type, thread, 1);

    ptx_reg_t finaldata;
    finaldata.pred = (data.u32 == 0) ? 1 : 0;  // setting zero-flag in predicate
    thread->set_operand_value(dst, finaldata, i_type, thread, pI);
  } else {
    data = thread->get_operand_value(src1, dst, i_type, thread, 1);

    thread->set_operand_value(dst, data, i_type, thread, pI);
  }
}

void mul24_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // src1_data = srcOperandModifiers(src1_data, src1, dst, i_type, thread);
  // src2_data = srcOperandModifiers(src2_data, src2, dst, i_type, thread);

  src1_data.mask_and(0, 0x00FFFFFF);
  src2_data.mask_and(0, 0x00FFFFFF);

  switch (i_type) {
    case S32_TYPE:
      if (src1_data.get_bit(23)) src1_data.mask_or(0xFFFFFFFF, 0xFF000000);
      if (src2_data.get_bit(23)) src2_data.mask_or(0xFFFFFFFF, 0xFF000000);
      data.s64 = src1_data.s64 * src2_data.s64;
      break;
    case U32_TYPE:
      data.u64 = src1_data.u64 * src2_data.u64;
      break;
    default:
      printf(
          "GPGPU-Sim PTX: Execution error - type mismatch with instruction\n");
      assert(0);
      break;
  }

  if (pI->is_hi()) {
    data.u64 = data.u64 >> 16;
    data.mask_and(0, 0xFFFFFFFF);
  } else if (pI->is_lo()) {
    data.mask_and(0, 0xFFFFFFFF);
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void mul_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  ptx_reg_t d, t;

  unsigned i_type = pI->get_type();
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  unsigned rounding_mode = pI->rounding_mode();

  switch (i_type) {
    case S16_TYPE:
      t.s32 = ((int)a.s16) * ((int)b.s16);
      if (pI->is_wide())
        d.s32 = t.s32;
      else if (pI->is_hi())
        d.s16 = (t.s32 >> 16);
      else if (pI->is_lo())
        d.s16 = t.s16;
      else
        assert(0);
      break;
    case S32_TYPE:
      t.s64 = ((long long)a.s32) * ((long long)b.s32);
      if (pI->is_wide())
        d.s64 = t.s64;
      else if (pI->is_hi())
        d.s32 = (t.s64 >> 32);
      else if (pI->is_lo())
        d.s32 = t.s32;
      else
        assert(0);
      break;
    case S64_TYPE:
      t.s64 = a.s64 * b.s64;
      assert(!pI->is_wide());
      // assert(!pI->is_hi());
      d.s64 = t.s64;
      break;
    case U16_TYPE:
      t.u32 = ((unsigned)a.u16) * ((unsigned)b.u16);
      if (pI->is_wide())
        d.u32 = t.u32;
      else if (pI->is_lo())
        d.u16 = t.u16;
      else if (pI->is_hi())
        d.u16 = (t.u32 >> 16);
      else
        assert(0);
      break;
    case U32_TYPE:
      t.u64 = ((unsigned long long)a.u32) * ((unsigned long long)b.u32);
      if (pI->is_wide())
        d.u64 = t.u64;
      else if (pI->is_lo())
        d.u32 = t.u32;
      else if (pI->is_hi())
        d.u32 = (t.u64 >> 32);
      else
        assert(0);
      break;
    case U64_TYPE:
      t.u64 = a.u64 * b.u64;
      assert(!pI->is_wide());
      assert(!pI->is_hi());
      if (pI->is_lo())
        d.u64 = t.u64;
      else
        assert(0);
      break;
    case F16_TYPE: {
      // assert(0);
      // break;
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }

      d.f16 = a.f16 * b.f16;

      if (pI->saturation_mode()) {
        if (d.f16 < 0)
          d.f16 = 0;
        else if (d.f16 > 1.0f)
          d.f16 = 1.0f;
      }
      fesetround(orig_rm);
      break;
    }
    case F32_TYPE: {
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }

      d.f32 = a.f32 * b.f32;

      if (pI->saturation_mode()) {
        if (d.f32 < 0)
          d.f32 = 0;
        else if (d.f32 > 1.0f)
          d.f32 = 1.0f;
      }
      fesetround(orig_rm);
      break;
    }
    case F64_TYPE:
    case FF64_TYPE: {
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }
      d.f64 = a.f64 * b.f64;
      if (pI->saturation_mode()) {
        if (d.f64 < 0)
          d.f64 = 0;
        else if (d.f64 > 1.0f)
          d.f64 = 1.0;
      }
      fesetround(orig_rm);
      break;
    }
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void neg_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned to_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, to_type, thread, 1);

  switch (to_type) {
    case S8_TYPE:
    case S16_TYPE:
    case S32_TYPE:
    case S64_TYPE:
      data.s64 = 0 - src1_data.s64;
      break;  // seems buggy, but not (just ignore higher bits)
    case U8_TYPE:
    case U16_TYPE:
    case U32_TYPE:
    case U64_TYPE:
      assert(0);
      break;
    case F16_TYPE:
      data.f16 = 0.0f - src1_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = 0.0f - src1_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = 0.0f - src1_data.f64;
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, to_type, thread, pI);
}

// nandn bitwise negates second operand then bitwise nands with the first
// operand
void nandn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = (~src1_data.pred & src2_data.pred);
  else
    data.u64 = ~(src1_data.u64 & ~src2_data.u64);

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

// norn bitwise negates first operand then bitwise ands with the second operand
void norn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(src1_data.pred & ~(src2_data.pred));
  else
    data.u64 = ~(src1_data.u64) & src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void not_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case PRED_TYPE:
      d.pred = (~(a.pred) & 0x000F);
      break;
    case B16_TYPE:
      d.u16 = ~a.u16;
      break;
    case B32_TYPE:
      d.u32 = ~a.u32;
      break;
    case B64_TYPE:
      d.u64 = ~a.u64;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void or_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(~(src1_data.pred) | ~(src2_data.pred));
  else
    data.u64 = src1_data.u64 | src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void orn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(~(src1_data.pred) | (src2_data.pred));
  else
    data.u64 = src1_data.u64 | ~src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void pmevent_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void popc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();

  unsigned i_type = pI->get_type();
  src_data = thread->get_operand_value(src, dst, i_type, thread, 1);

  switch (i_type) {
    case B32_TYPE: {
      std::bitset<32> mask(src_data.u32);
      data.u32 = mask.count();
    } break;
    case B64_TYPE: {
      std::bitset<64> mask(src_data.u64);
      data.u32 = mask.count();
    } break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}
void prefetch_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void prefetchu_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

int prmt_mode_present(int mode) {
  int returnval = 0;
  switch (mode) {
    case PRMT_F4E_MODE:
    case PRMT_B4E_MODE:
    case PRMT_RC8_MODE:
    case PRMT_RC16_MODE:
    case PRMT_ECL_MODE:
    case PRMT_ECR_MODE:
      returnval = 1;
      break;
    default:
      break;
  }
  return returnval;
}
int read_byte(int mode, int control, int d_sel_index, signed long long value) {
  int returnval = 0;
  int prmt_f4e_mode[4][4] = {
      {0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}};
  int prmt_b4e_mode[4][4] = {
      {0, 7, 6, 5}, {1, 0, 7, 6}, {2, 1, 0, 7}, {3, 2, 1, 0}};
  int prmt_rc8_mode[4][4] = {
      {0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}};
  int prmt_ecl_mode[4][4] = {
      {0, 1, 2, 3}, {1, 1, 2, 3}, {2, 2, 2, 3}, {3, 3, 3, 3}};
  int prmt_ecr_mode[4][4] = {
      {0, 0, 0, 0}, {0, 1, 1, 1}, {0, 1, 2, 2}, {0, 1, 2, 3}};
  int prmt_rc16_mode[4][4] = {
      {0, 1, 0, 1}, {2, 3, 2, 3}, {0, 1, 0, 1}, {2, 3, 2, 3}};

  if (!prmt_mode_present(mode)) {
    if (control & 0x8) {
      returnval = 0xff;
    } else {
      returnval = (value >> (8 * control)) & 0xff;
    }
  } else {
    switch (mode) {
      case PRMT_F4E_MODE:
        returnval = prmt_f4e_mode[control][d_sel_index];
        break;
      case PRMT_B4E_MODE:
        returnval = prmt_b4e_mode[control][d_sel_index];
        break;
      case PRMT_RC8_MODE:
        returnval = prmt_rc8_mode[control][d_sel_index];
        break;
      case PRMT_ECL_MODE:
        returnval = prmt_ecl_mode[control][d_sel_index];
        break;
      case PRMT_ECR_MODE:
        returnval = prmt_ecr_mode[control][d_sel_index];
        break;
      case PRMT_RC16_MODE:
        returnval = prmt_rc16_mode[control][d_sel_index];
        break;
        // Change the default from printing "ERROR" to just asserting
      default:
        assert(false);
    }
  }
  return (returnval << 8 * d_sel_index);
}

void prmt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, src3_data, tmpdata, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  unsigned mode = pI->prmt_op();
  unsigned i_type = pI->get_type();

  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);

  tmpdata.s64 = src1_data.s32 | (src2_data.s64 << 32);
  int ctl[4];

  if (!prmt_mode_present(mode)) {
    ctl[0] = (src3_data.s32 >> 0) & 0xf;
    ctl[1] = (src3_data.s32 >> 4) & 0xf;
    ctl[2] = (src3_data.s32 >> 8) & 0xf;
    ctl[3] = (src3_data.s32 >> 12) & 0xf;
  } else {
    ctl[0] = ctl[1] = ctl[2] = ctl[3] = (src3_data.s32 >> 0) & 0x3;
  }

  data.s32 = 0;
  data.s32 = data.s32 | read_byte(mode, ctl[0], 0, tmpdata.s64);  // First
                                                                  // byte-0
  data.s32 =
      data.s32 | read_byte(mode, ctl[1], 1, tmpdata.s64);  // Second byte-1
  data.s32 = data.s32 | read_byte(mode, ctl[2], 2, tmpdata.s64);  // Third
                                                                  // byte-2
  data.s32 =
      data.s32 | read_byte(mode, ctl[3], 3, tmpdata.s64);  // Fourth byte-3

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void rcp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      data.f32 = 1.0f / src1_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = 1.0f / src1_data.f64;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void red_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void rem_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case S32_TYPE:
      data.s32 = src1_data.s32 % src2_data.s32;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 % src2_data.s64;
      break;
    case U32_TYPE:
      data.u32 = src1_data.u32 % src2_data.u32;
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 % src2_data.u64;
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void ret_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  bool empty = thread->callstack_pop();
  if (empty) {
    thread->set_done();
    thread->exitCore();
    thread->registerExit();
  }
}

// Ptxplus version of ret instruction.
void retp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  bool empty = thread->callstack_pop_plus();
  if (empty) {
    thread->set_done();
    thread->exitCore();
    thread->registerExit();
  }
}

void rsqrt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      if (a.f32 < 0) {
        d.u64 = 0;
        d.u64 = 0x7fc00000;  // NaN
      } else if (a.f32 == 0) {
        d.u64 = 0;
        d.u32 = 0x7f800000;  // Inf
      } else
        d.f32 = cuda_math::__internal_accurate_fdividef(1.0f, sqrtf(a.f32));
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (a.f32 < 0) {
        d.u64 = 0;
        d.u32 = 0x7fc00000;  // NaN
        float x = d.f32;
        d.f64 = (double)x;
      } else if (a.f32 == 0) {
        d.u64 = 0;
        d.u32 = 0x7f800000;  // Inf
        float x = d.f32;
        d.f64 = (double)x;
      } else
        d.f64 = 1.0 / sqrt(a.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

#define SAD(d, a, b, c) d = c + ((a < b) ? (b - a) : (a - b))

void sad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, c, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
      SAD(d.u16, a.u16, b.u16, c.u16);
      break;
    case U32_TYPE:
      SAD(d.u32, a.u32, b.u32, c.u32);
      break;
    case U64_TYPE:
      SAD(d.u64, a.u64, b.u64, c.u64);
      break;
    case S16_TYPE:
      SAD(d.s16, a.s16, b.s16, c.s16);
      break;
    case S32_TYPE:
      SAD(d.s32, a.s32, b.s32, c.s32);
      break;
    case S64_TYPE:
      SAD(d.s64, a.s64, b.s64, c.s64);
      break;
    case F32_TYPE:
      SAD(d.f32, a.f32, b.f32, c.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      SAD(d.f64, a.f64, b.f64, c.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void selp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  ptx_reg_t a, b, c, d;

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  // predicate value was changed so the lowest bit being set means the zero flag
  // is set. As a result, the value of c.pred must be inverted to get proper
  // behavior
  d = (!(c.pred & 0x0001)) ? a : b;

  thread->set_operand_value(dst, d, PRED_TYPE, thread, pI);
}

bool isFloat(int type) {
  switch (type) {
    case F16_TYPE:
    case F32_TYPE:
    case F64_TYPE:
    case FF64_TYPE:
      return true;
    default:
      return false;
  }
}

bool CmpOp(int type, ptx_reg_t a, ptx_reg_t b, unsigned cmpop) {
  bool t = false;

  switch (type) {
    case B16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u16 == b.u16);
          break;
        case NE_OPTION:
          t = (a.u16 != b.u16);
          break;
        default:
          assert(0);
      }

    case B32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u32 == b.u32);
          break;
        case NE_OPTION:
          t = (a.u32 != b.u32);
          break;
        default:
          assert(0);
      }
    case B64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u64 == b.u64);
          break;
        case NE_OPTION:
          t = (a.u64 != b.u64);
          break;
        default:
          assert(0);
      }
      break;
    case S8_TYPE:
    case S16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s16 == b.s16);
          break;
        case NE_OPTION:
          t = (a.s16 != b.s16);
          break;
        case LT_OPTION:
          t = (a.s16 < b.s16);
          break;
        case LE_OPTION:
          t = (a.s16 <= b.s16);
          break;
        case GT_OPTION:
          t = (a.s16 > b.s16);
          break;
        case GE_OPTION:
          t = (a.s16 >= b.s16);
          break;
        default:
          assert(0);
      }
      break;
    case S32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s32 == b.s32);
          break;
        case NE_OPTION:
          t = (a.s32 != b.s32);
          break;
        case LT_OPTION:
          t = (a.s32 < b.s32);
          break;
        case LE_OPTION:
          t = (a.s32 <= b.s32);
          break;
        case GT_OPTION:
          t = (a.s32 > b.s32);
          break;
        case GE_OPTION:
          t = (a.s32 >= b.s32);
          break;
        default:
          assert(0);
      }
      break;
    case S64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s64 == b.s64);
          break;
        case NE_OPTION:
          t = (a.s64 != b.s64);
          break;
        case LT_OPTION:
          t = (a.s64 < b.s64);
          break;
        case LE_OPTION:
          t = (a.s64 <= b.s64);
          break;
        case GT_OPTION:
          t = (a.s64 > b.s64);
          break;
        case GE_OPTION:
          t = (a.s64 >= b.s64);
          break;
        default:
          assert(0);
      }
      break;
    case U8_TYPE:
    case U16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u16 == b.u16);
          break;
        case NE_OPTION:
          t = (a.u16 != b.u16);
          break;
        case LT_OPTION:
          t = (a.u16 < b.u16);
          break;
        case LE_OPTION:
          t = (a.u16 <= b.u16);
          break;
        case GT_OPTION:
          t = (a.u16 > b.u16);
          break;
        case GE_OPTION:
          t = (a.u16 >= b.u16);
          break;
        case LO_OPTION:
          t = (a.u16 < b.u16);
          break;
        case LS_OPTION:
          t = (a.u16 <= b.u16);
          break;
        case HI_OPTION:
          t = (a.u16 > b.u16);
          break;
        case HS_OPTION:
          t = (a.u16 >= b.u16);
          break;
        default:
          assert(0);
      }
      break;
    case U32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u32 == b.u32);
          break;
        case NE_OPTION:
          t = (a.u32 != b.u32);
          break;
        case LT_OPTION:
          t = (a.u32 < b.u32);
          break;
        case LE_OPTION:
          t = (a.u32 <= b.u32);
          break;
        case GT_OPTION:
          t = (a.u32 > b.u32);
          break;
        case GE_OPTION:
          t = (a.u32 >= b.u32);
          break;
        case LO_OPTION:
          t = (a.u32 < b.u32);
          break;
        case LS_OPTION:
          t = (a.u32 <= b.u32);
          break;
        case HI_OPTION:
          t = (a.u32 > b.u32);
          break;
        case HS_OPTION:
          t = (a.u32 >= b.u32);
          break;
        default:
          assert(0);
      }
      break;
    case U64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u64 == b.u64);
          break;
        case NE_OPTION:
          t = (a.u64 != b.u64);
          break;
        case LT_OPTION:
          t = (a.u64 < b.u64);
          break;
        case LE_OPTION:
          t = (a.u64 <= b.u64);
          break;
        case GT_OPTION:
          t = (a.u64 > b.u64);
          break;
        case GE_OPTION:
          t = (a.u64 >= b.u64);
          break;
        case LO_OPTION:
          t = (a.u64 < b.u64);
          break;
        case LS_OPTION:
          t = (a.u64 <= b.u64);
          break;
        case HI_OPTION:
          t = (a.u64 > b.u64);
          break;
        case HS_OPTION:
          t = (a.u64 >= b.u64);
          break;
        default:
          assert(0);
      }
      break;
    case F16_TYPE:
      assert(0);
      break;
    case F32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.f32 == b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case NE_OPTION:
          t = (a.f32 != b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case LT_OPTION:
          t = (a.f32 < b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case LE_OPTION:
          t = (a.f32 <= b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case GT_OPTION:
          t = (a.f32 > b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case GE_OPTION:
          t = (a.f32 >= b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case EQU_OPTION:
          t = (a.f32 == b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case NEU_OPTION:
          t = (a.f32 != b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case LTU_OPTION:
          t = (a.f32 < b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case LEU_OPTION:
          t = (a.f32 <= b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case GTU_OPTION:
          t = (a.f32 > b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case GEU_OPTION:
          t = (a.f32 >= b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case NUM_OPTION:
          t = !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case NAN_OPTION:
          t = isNaN(a.f32) || isNaN(b.f32);
          break;
        default:
          assert(0);
      }
      break;
    case F64_TYPE:
    case FF64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.f64 == b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case NE_OPTION:
          t = (a.f64 != b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case LT_OPTION:
          t = (a.f64 < b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case LE_OPTION:
          t = (a.f64 <= b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case GT_OPTION:
          t = (a.f64 > b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case GE_OPTION:
          t = (a.f64 >= b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case EQU_OPTION:
          t = (a.f64 == b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case NEU_OPTION:
          t = (a.f64 != b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case LTU_OPTION:
          t = (a.f64 < b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case LEU_OPTION:
          t = (a.f64 <= b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case GTU_OPTION:
          t = (a.f64 > b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case GEU_OPTION:
          t = (a.f64 >= b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case NUM_OPTION:
          t = !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case NAN_OPTION:
          t = isNaN(a.f64) || isNaN(b.f64);
          break;
        default:
          assert(0);
      }
      break;
    default:
      assert(0);
      break;
  }

  return t;
}

void setp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b;

  int t = 0;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  assert(pI->get_num_operands() <
         4);  // or need to deal with "c" operand / boolOp

  unsigned type = pI->get_type();
  unsigned cmpop = pI->get_cmpop();
  a = thread->get_operand_value(src1, dst, type, thread, 1);
  b = thread->get_operand_value(src2, dst, type, thread, 1);

  t = CmpOp(type, a, b, cmpop);

  ptx_reg_t data;

  // the way ptxplus handles the zero flag, 1 = false and 0 = true
  data.pred =
      (t ==
       0);  // inverting predicate since ptxplus uses "1" for a set zero flag

  thread->set_operand_value(dst, data, PRED_TYPE, thread, pI);
}

void set_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b;

  int t = 0;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  assert(pI->get_num_operands() <
         4);  // or need to deal with "c" operand / boolOp

  unsigned src_type = pI->get_type2();
  unsigned cmpop = pI->get_cmpop();

  a = thread->get_operand_value(src1, dst, src_type, thread, 1);
  b = thread->get_operand_value(src2, dst, src_type, thread, 1);

  // Take abs of first operand if needed
  if (pI->is_abs()) {
    switch (src_type) {
      case S16_TYPE:
        a.s16 = my_abs(a.s16);
        break;
      case S32_TYPE:
        a.s32 = my_abs(a.s32);
        break;
      case S64_TYPE:
        a.s64 = my_abs(a.s64);
        break;
      case U16_TYPE:
        a.u16 = a.u16;
        break;
      case U32_TYPE:
        a.u32 = my_abs(a.u32);
        break;
      case U64_TYPE:
        a.u64 = my_abs(a.u64);
        break;
      case F32_TYPE:
        a.f32 = my_abs(a.f32);
        break;
      case F64_TYPE:
      case FF64_TYPE:
        a.f64 = my_abs(a.f64);
        break;
      default:
        printf("Execution error: type mismatch with instruction\n");
        assert(0);
        break;
    }
  }

  t = CmpOp(src_type, a, b, cmpop);

  ptx_reg_t data;
  if (isFloat(pI->get_type())) {
    data.f32 = (t != 0) ? 1.0f : 0.0f;
  } else {
    data.u32 = (t != 0) ? 0xFFFFFFFF : 0;
  }

  thread->set_operand_value(dst, data, pI->get_type(), thread, pI);
}

void shfl_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst) {
  unsigned i_type = pI->get_type();
  int tid;

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  ptx_thread_info *thread = core->get_thread_info()[tid];
  ptx_warp_info *warp_info = thread->m_warp_info;
  int lane = warp_info->get_done_threads();
  thread = core->get_thread_info()[tid + lane];

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  int bval = (thread->get_operand_value(src2, dst, i_type, thread, 1)).u32;
  int cval = (thread->get_operand_value(src3, dst, i_type, thread, 1)).u32;
  int mask = cval >> 8;
  bval &= 0x1F;
  cval &= 0x1F;

  int maxLane = (lane & mask) | (cval & ~mask);
  int minLane = lane & mask;

  int src_idx;
  unsigned p;
  switch (pI->shfl_op()) {
    case UP_OPTION:
      src_idx = lane - bval;
      p = (src_idx >= maxLane);
      break;
    case DOWN_OPTION:
      src_idx = lane + bval;
      p = (src_idx <= maxLane);
      break;
    case BFLY_OPTION:
      src_idx = lane ^ bval;
      p = (src_idx <= maxLane);
      break;
    case IDX_OPTION:
      src_idx = minLane | (bval & ~mask);
      p = (src_idx <= maxLane);
      break;
    default:
      printf("GPGPU-Sim PTX: ERROR: Invalid shfl option\n");
      assert(0);
      break;
  }
  // copy from own lane
  if (!p) src_idx = lane;

  // copy input from lane src_idx
  ptx_reg_t data;
  if (inst.active(src_idx)) {
    ptx_thread_info *source = core->get_thread_info()[tid + src_idx];
    data = source->get_operand_value(src1, dst, i_type, source, 1);
  } else {
    printf(
        "GPGPU-Sim PTX: WARNING: shfl input value unpredictable for inactive "
        "threads in a warp\n");
    data.u32 = 0;
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);

  /*
  TODO: deal with predicates appropriately using the following pseudocode:
  if (!isGuardPredicateTrue(src_idx)) {
          printf("GPGPU-Sim PTX: WARNING: shfl input value unpredictable for
  predicated-off threads in a warp\n");
  }
  if (dest predicate selected) data.pred = p;
  */

  // keep track of the number of threads that have executed in the warp
  warp_info->inc_done_threads();
  if (warp_info->get_done_threads() == inst.active_count()) {
    warp_info->reset_done_threads();
  }
}

void shf_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, c, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  // Only b32 is allowed
  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  if (i_type != B32_TYPE)
    printf("Only the b32 data_type is allowed per the ISA\n");

  unsigned clamp_mode = pI->clamp_mode();
  unsigned n = c.u32 & 0x1f;
  if (clamp_mode) {
    if (c.u32 < 32)
      n = c;
    else
      n = 32;
  }
  if (pI->left_mode())
    d.u32 = (b.u32 << n) | (a.u32 >> (32 - n));
  else
    d.u32 = (b.u32 << (32 - n)) | (a.u32 >> n);

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void shl_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case B16_TYPE:
    case U16_TYPE:
      if (b.u16 >= 16)
        d.u16 = 0;
      else
        d.u16 = (unsigned short)((a.u16 << b.u16) & 0xFFFF);
      break;
    case B32_TYPE:
    case U32_TYPE:
      if (b.u32 >= 32)
        d.u32 = 0;
      else
        d.u32 = (unsigned)((a.u32 << b.u32) & 0xFFFFFFFF);
      break;
    case B64_TYPE:
    case U64_TYPE:
      if (b.u32 >= 64)
        d.u64 = 0;
      else
        d.u64 = (a.u64 << b.u64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void shr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
    case B16_TYPE:
      if (b.u16 < 16)
        d.u16 = (unsigned short)((a.u16 >> b.u16) & 0xFFFF);
      else
        d.u16 = 0;
      break;
    case U32_TYPE:
    case B32_TYPE:
      if (b.u32 < 32)
        d.u32 = (unsigned)((a.u32 >> b.u32) & 0xFFFFFFFF);
      else
        d.u32 = 0;
      break;
    case U64_TYPE:
    case B64_TYPE:
      if (b.u32 < 64)
        d.u64 = (a.u64 >> b.u64);
      else
        d.u64 = 0;
      break;
    case S16_TYPE:
      if (b.u16 < 16)
        d.s64 = (a.s16 >> b.s16);
      else {
        if (a.s16 < 0) {
          d.s64 = -1;
        } else {
          d.s64 = 0;
        }
      }
      break;
    case S32_TYPE:
      if (b.u32 < 32)
        d.s64 = (a.s32 >> b.s32);
      else {
        if (a.s32 < 0) {
          d.s64 = -1;
        } else {
          d.s64 = 0;
        }
      }
      break;
    case S64_TYPE:
      if (b.u64 < 64)
        d.s64 = (a.s64 >> b.u64);
      else {
        if (a.s64 < 0) {
          if (b.s32 < 0) {
            d.u64 = -1;
            d.s32 = 0;
          } else {
            d.s64 = -1;
          }
        } else {
          d.s64 = 0;
        }
      }
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void sin_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      d.f32 = sin(a.f32);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void slct_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  ptx_reg_t a, b, c, d;

  unsigned i_type = pI->get_type();
  unsigned c_type = pI->get_type2();
  bool t = false;
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, c_type, thread, 1);

  switch (c_type) {
    case S32_TYPE:
      t = c.s32 >= 0;
      break;
    case F32_TYPE:
      t = c.f32 >= 0;
      break;
    default:
      assert(0);
  }

  switch (i_type) {
    case B16_TYPE:
    case S16_TYPE:
    case U16_TYPE:
      d.u16 = t ? a.u16 : b.u16;
      break;
    case F32_TYPE:
    case B32_TYPE:
    case S32_TYPE:
    case U32_TYPE:
      d.u32 = t ? a.u32 : b.u32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
    case B64_TYPE:
    case S64_TYPE:
    case U64_TYPE:
      d.u64 = t ? a.u64 : b.u64;
      break;
    default:
      assert(0);
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void sqrt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      if (a.f32 < 0)
        d.f32 = nanf("");
      else
        d.f32 = sqrt(a.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (a.f64 < 0)
        d.f64 = nan("");
      else
        d.f64 = sqrt(a.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void sst_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_instruction *cpI = const_cast<ptx_instruction *>(pI);  // constant
  const operand_info &dst = cpI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  unsigned type = pI->get_type();
  ptx_reg_t dst_data = thread->get_operand_value(dst, dst, type, thread, 1);
  ptx_reg_t src1_data = thread->get_operand_value(src1, src1, type, thread, 1);
  ptx_reg_t src2_data = thread->get_operand_value(src2, src1, type, thread, 1);
  ptx_reg_t src3_data = thread->get_operand_value(src3, src1, type, thread, 1);
  memory_space_t space = pI->get_space();
  memory_space *mem = NULL;
  addr_t addr =
      src2_data.u32 * 4;  // this assumes sstarr memory starts at address 0
  ptx_cta_info *cta_info = thread->m_cta_info;

  decode_space(space, thread, src1, mem, addr);

  size_t size;
  int t;
  type_info_key::type_decode(type, size, t);

  // store data in sstarr memory
  mem->write(addr, size / 8, &src3_data.s64, thread, pI);

  // sync threads
  cpI->set_bar_id(16);  // use 16 for sst because bar uses an int from 0-15

  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = space;
  thread->m_last_dram_callback.function = bar_callback;
  thread->m_last_dram_callback.instruction = cpI;

  // the last thread that executes loads all of the data back from sstarr memory
  int NUM_THREADS = cta_info->num_threads();
  cta_info->inc_bar_threads();
  if (NUM_THREADS == cta_info->get_bar_threads()) {
    unsigned offset = 0;
    addr = 0;
    ptx_reg_t data;
    float sstarr_fdata[NUM_THREADS];
    signed long long sstarr_ldata[NUM_THREADS];
    // loop through all of the threads
    for (int tid = 0; tid < NUM_THREADS; tid++) {
      data.u64 = 0;
      mem->read(addr + (tid * 4), size / 8, &data.s64);
      sstarr_fdata[tid] = data.f32;
      sstarr_ldata[tid] = data.s64;
    }

    // squeeze the zeros out of the array and store data back into original
    // array
    mem = NULL;
    addr = src1_data.u32;
    space.set_type(global_space);
    decode_space(space, thread, src1, mem, addr);
    // store nonzero entries and indices
    for (int tid = 0; tid < NUM_THREADS; tid++) {
      if (sstarr_fdata[tid] != 0) {
        float ftid = (float)tid;
        mem->write(addr + (offset * 4), size / 8, &sstarr_ldata[tid], thread,
                   pI);
        mem->write(addr + ((NUM_THREADS + offset) * 4), size / 8, &ftid, thread,
                   pI);
        offset++;
      }
    }
    // store the number of nonzero elements in the array
    data = thread->get_operand_value(src1, dst, type, thread, 1);
    data.s64 += 4 * (offset - 1);
    thread->set_operand_value(dst, data, type, thread, pI);

    // fill the rest of the array with zeros (dst should always have a 0 in it)
    while (offset < NUM_THREADS) {
      mem->write(addr + (offset * 4), size / 8, &dst_data.s64, thread, pI);
      offset++;
    }

    cta_info->reset_bar_threads();
    thread->m_last_effective_address = addr + (NUM_THREADS - 1) * 4;
    thread->m_last_memory_space = space;
  }
}

void ssy_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // printf("Execution Warning: unimplemented ssy instruction is treated as a
  // nop\n");
  // TODO: add implementation
}

void st_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();  // may be scalar or vector of regs
  unsigned type = pI->get_type();
  ptx_reg_t addr_reg = thread->get_operand_value(dst, dst, type, thread, 1);
  ptx_reg_t data;
  memory_space_t space = pI->get_space();
  unsigned vector_spec = pI->get_vector();

  memory_space *mem = NULL;
  addr_t addr = addr_reg.u32;

  decode_space(space, thread, dst, mem, addr);

  size_t size;
  int t;
  type_info_key::type_decode(type, size, t);

  if (!vector_spec) {
    data = thread->get_operand_value(src1, dst, type, thread, 1);
    mem->write(addr, size / 8, &data.s64, thread, pI);
  } else {
    if (vector_spec == V2_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[2];
      thread->get_vector_operand_values(src1, ptx_regs, 2);
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      delete[] ptx_regs;
    }
    if (vector_spec == V3_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[3];
      thread->get_vector_operand_values(src1, ptx_regs, 3);
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      mem->write(addr + 2 * size / 8, size / 8, &ptx_regs[2].s64, thread, pI);
      delete[] ptx_regs;
    }
    if (vector_spec == V4_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[4];
      thread->get_vector_operand_values(src1, ptx_regs, 4);
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      mem->write(addr + 2 * size / 8, size / 8, &ptx_regs[2].s64, thread, pI);
      mem->write(addr + 3 * size / 8, size / 8, &ptx_regs[3].s64, thread, pI);
      delete[] ptx_regs;
    }
  }
  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = space;
}

void sub_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;
  int overflow = 0;
  int carry = 0;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // performs addition. Sets carry and overflow if needed.
  // the constant is added in during subtraction so the carry bit is set
  // properly.
  switch (i_type) {
    case S8_TYPE:
      data.s64 = (src1_data.s64 & 0xFF) - (src2_data.s64 & 0xFF) + 0x100;
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) != 0) {
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;
      }
      carry = (data.s32 & 0x100) >> 8;
      break;
    case S16_TYPE:
      data.s64 = (src1_data.s64 & 0xFFFF) - (src2_data.s64 & 0xFFFF) + 0x10000;
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) != 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.s32 & 0x10000) >> 16;
      break;
    case S32_TYPE:
      data.s64 = (src1_data.s64 & 0xFFFFFFFF) - (src2_data.s64 & 0xFFFFFFFF) +
                 0x100000000;
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) != 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = ((data.u64) >> 32) & 0x0001;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 - src2_data.s64;
      break;
    case B8_TYPE:
    case U8_TYPE:
      data.u64 = (src1_data.u64 & 0xFF) - (src2_data.u64 & 0xFF) + 0x100;
      carry = (data.u64 & 0x100) >> 8;
      break;
    case B16_TYPE:
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) - (src2_data.u64 & 0xFFFF) + 0x10000;
      carry = (data.u64 & 0x10000) >> 16;
      break;
    case B32_TYPE:
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) - (src2_data.u64 & 0xFFFFFFFF) +
                 0x100000000;
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case B64_TYPE:
    case U64_TYPE:
      data.u64 = src1_data.u64 - src2_data.u64;
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 - src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 - src2_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 - src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);
}

void nop_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // Do nothing
}

void subc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void suld_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void sured_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void sust_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void suq_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

union intfloat {
  int a;
  float b;
};

float reduce_precision(float x, unsigned bits) {
  intfloat tmp;
  tmp.b = x;
  int v = tmp.a;
  int man = v & ((1 << 23) - 1);
  int mask = ((1 << bits) - 1) << (23 - bits);
  int nv = (v & ((-1) - ((1 << 23) - 1))) | (mask & man);
  tmp.a = nv;
  float result = tmp.b;
  return result;
}

unsigned wrap(unsigned x, unsigned y, unsigned mx, unsigned my,
              size_t elem_size) {
  unsigned nx = (mx + x) % mx;
  unsigned ny = (my + y) % my;
  return nx + mx * ny;
}

unsigned clamp(unsigned x, unsigned y, unsigned mx, unsigned my,
               size_t elem_size) {
  unsigned nx = x;
  while (nx >= mx) nx -= elem_size;
  unsigned ny = (y >= my) ? my - 1 : y;
  return nx + mx * ny;
}

typedef unsigned (*texAddr_t)(unsigned x, unsigned y, unsigned mx, unsigned my,
                              size_t elem_size);
float tex_linf_sampling(memory_space *mem, unsigned tex_array_base, int x,
                        int y, unsigned int width, unsigned int height,
                        size_t elem_size, float alpha, float beta,
                        texAddr_t b_lim) {
  float Tij;
  float Ti1j;
  float Tij1;
  float Ti1j1;

  mem->read(tex_array_base + b_lim(x, y, width, height, elem_size), 4, &Tij);
  mem->read(tex_array_base + b_lim(x + elem_size, y, width, height, elem_size),
            4, &Ti1j);
  mem->read(tex_array_base + b_lim(x, y + 1, width, height, elem_size), 4,
            &Tij1);
  mem->read(
      tex_array_base + b_lim(x + elem_size, y + 1, width, height, elem_size), 4,
      &Ti1j1);

  float sample = (1 - alpha) * (1 - beta) * Tij + alpha * (1 - beta) * Ti1j +
                 (1 - alpha) * beta * Tij1 + alpha * beta * Ti1j1;

  return sample;
}

float textureNormalizeElementSigned(int element, int bits) {
  if (bits) {
    int maxN = (1 << bits) - 1;
    // removing upper bits
    element &= maxN;
    // normalizing the number to [-1.0,1.0]
    maxN >>= 1;
    float output = (float)element / maxN;
    if (output < -1.0f) output = -1.0f;
    return output;
  } else {
    return 0.0f;
  }
}

float textureNormalizeElementUnsigned(unsigned int element, int bits) {
  if (bits) {
    unsigned int maxN = (1 << bits) - 1;
    // removing upper bits and normalizing the number to [0.0,1.0]
    return (float)(element & maxN) / maxN;
  } else {
    return 0.0f;
  }
}

void textureNormalizeOutput(const struct cudaChannelFormatDesc &desc,
                            ptx_reg_t &datax, ptx_reg_t &datay,
                            ptx_reg_t &dataz, ptx_reg_t &dataw) {
  if (desc.f == cudaChannelFormatKindSigned) {
    datax.f32 = textureNormalizeElementSigned(datax.s32, desc.x);
    datay.f32 = textureNormalizeElementSigned(datay.s32, desc.y);
    dataz.f32 = textureNormalizeElementSigned(dataz.s32, desc.z);
    dataw.f32 = textureNormalizeElementSigned(dataw.s32, desc.w);
  } else if (desc.f == cudaChannelFormatKindUnsigned) {
    datax.f32 = textureNormalizeElementUnsigned(datax.u32, desc.x);
    datay.f32 = textureNormalizeElementUnsigned(datay.u32, desc.y);
    dataz.f32 = textureNormalizeElementUnsigned(dataz.u32, desc.z);
    dataw.f32 = textureNormalizeElementUnsigned(dataw.u32, desc.w);
  } else {
    assert(0 &&
           "Undefined texture read mode: cudaReadModeNormalizedFloat expect "
           "integer elements");
  }
}

void tex_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
#if (CUDART_VERSION <= 1200)
  unsigned dimension = pI->dimension();
  const operand_info &dst =
      pI->dst();  // the registers to which fetched texel will be placed
  const operand_info &src1 = pI->src1();  // the name of the texture
  const operand_info &src2 =
      pI->src2();  // the vector registers containing coordinates of the texel
                   // to be fetched

  std::string texname = src1.name();
  // If indirect access, use register's value as address
  // to find the symbol
  if (src1.is_reg()) {
    ptx_reg_t src1_data =
        thread->get_operand_value(src1, dst, pI->get_type(), thread, 1);
    addr_t sym_addr = src1_data.u64;
    symbol *texRef = thread->get_symbol_table()->lookup_by_addr(sym_addr);
    assert(texRef != NULL);
    texname = texRef->name();
  }

  unsigned to_type = pI->get_type();
  unsigned c_type = pI->get_type2();
  fflush(stdout);
  ptx_reg_t data1, data2, data3, data4;
  if (!thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs)
    thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs = new ptx_reg_t[4];
  unsigned nelem = src2.get_vect_nelem();
  thread->get_vector_operand_values(
      src2, thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs,
      nelem);  // ptx_reg should be 4 entry vector type...coordinates into
               // texture
  /*
    For programs with many streams, textures can be bound and unbound
    asynchronously.  This means we need to use the kernel's "snapshot" of
    the state of the texture mappings when it was launched (so that we
    don't try to access the incorrect texture mapping if it's been updated,
    or that we don't access a mapping that has been unbound).
  */
  gpgpu_t *gpu = thread->get_gpu();
  kernel_info_t &k = thread->get_kernel();
  const struct textureReference *texref = gpu->get_texref(texname);
  const struct cudaArray *cuArray = k.get_texarray(texname);
  const struct textureInfo *texInfo = k.get_texinfo(texname);
  const struct textureReferenceAttr *texAttr = gpu->get_texattr(texname);

  // assume always 2D f32 input
  // access array with src2 coordinates
  memory_space *mem = thread->get_global_memory();
  float x_f32, y_f32;
  size_t size;
  int t;
  unsigned tex_array_base;
  unsigned int width = 0, height = 0;
  int x = 0;
  int y = 0;
  unsigned tex_array_index;
  float alpha = 0, beta = 0;

  type_info_key::type_decode(to_type, size, t);
  tex_array_base = cuArray->devPtr32;

  switch (dimension) {
    case GEOM_MODIFIER_1D:
      width = cuArray->width;
      height = cuArray->height;
      if (texref->normalized) {
        assert(c_type == F32_TYPE);
        x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          x_f32 = (x_f32 > 1.0) ? 1.0 : x_f32;
          x_f32 = (x_f32 < 0.0) ? 0.0 : x_f32;
        } else if (texref->addressMode[0] == cudaAddressModeWrap) {
          x_f32 = x_f32 - floor(x_f32);
        }

        if (texref->filterMode == cudaFilterModeLinear) {
          float xb = x_f32 * width - 0.5;
          alpha = xb - floor(xb);
          alpha = reduce_precision(alpha, 9);
          beta = 0.0;

          x = (int)floor(xb);
          y = 0;
        } else {
          x = (int)floor(x_f32 * width);
          y = 0;
        }
      } else {
        switch (c_type) {
          case S32_TYPE:
            x = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].s32;
            assert(texref->filterMode == cudaFilterModePoint);
            break;
          case F32_TYPE:
            x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
            alpha = x_f32 -
                    floor(x_f32);  // offset into subtexel (for linear sampling)
            x = (int)x_f32;
            break;
          default:
            assert(0 && "Unsupported texture coordinate type.");
        }
        // handle texture fetch that exceeded boundaries
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          x = (x > width - 1) ? (width - 1) : x;
          x = (x < 0) ? 0 : x;
        } else if (texref->addressMode[0] == cudaAddressModeWrap) {
          x = x % width;
        }
      }
      width *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
                cuArray->desc.z) /
               8;
      x *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
            cuArray->desc.z) /
           8;
      tex_array_index = tex_array_base + x;

      break;
    case GEOM_MODIFIER_2D:
      width = cuArray->width;
      height = cuArray->height;
      if (texref->normalized) {
        x_f32 = reduce_precision(
            thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32, 16);
        y_f32 = reduce_precision(
            thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[1].f32, 15);

        if (texref->addressMode[0]) {  // clamp
          if (x_f32 < 0) x_f32 = 0;
          if (x_f32 >= 1) x_f32 = 1 - 1 / x_f32;
        } else {  // wrap
          x_f32 = x_f32 - floor(x_f32);
        }
        if (texref->addressMode[1]) {  // clamp
          if (y_f32 < 0) y_f32 = 0;
          if (y_f32 >= 1) y_f32 = 1 - 1 / y_f32;
        } else {  // wrap
          y_f32 = y_f32 - floor(y_f32);
        }

        if (texref->filterMode == cudaFilterModeLinear) {
          float xb = x_f32 * width - 0.5;
          float yb = y_f32 * height - 0.5;
          alpha = xb - floor(xb);
          beta = yb - floor(yb);
          alpha = reduce_precision(alpha, 9);
          beta = reduce_precision(beta, 9);

          x = (int)floor(xb);
          y = (int)floor(yb);
        } else {
          x = (int)floor(x_f32 * width);
          y = (int)floor(y_f32 * height);
        }
      } else {
        x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
        y_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[1].f32;

        alpha = x_f32 - floor(x_f32);
        beta = y_f32 - floor(y_f32);

        x = (int)x_f32;
        y = (int)y_f32;
        if (texref->addressMode[0]) {  // clamp
          if (x < 0) x = 0;
          if (x >= (int)width) x = width - 1;
        } else {  // wrap
          x = x % width;
          if (x < 0) x *= -1;
        }
        if (texref->addressMode[1]) {  // clamp
          if (y < 0) y = 0;
          if (y >= (int)height) y = height - 1;
        } else {  // wrap
          y = y % height;
          if (y < 0) y *= -1;
        }
      }

      width *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
                cuArray->desc.z) /
               8;
      x *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
            cuArray->desc.z) /
           8;
      tex_array_index = tex_array_base + (x + width * y);
      break;
    default:
      assert(0);
      break;
  }
  switch (to_type) {
    case U8_TYPE:
    case U16_TYPE:
    case U32_TYPE:
    case B8_TYPE:
    case B16_TYPE:
    case B32_TYPE:
    case S8_TYPE:
    case S16_TYPE:
    case S32_TYPE: {
      unsigned long long elementOffset = 0;  // offset into the next element
      mem->read(tex_array_index, cuArray->desc.x / 8, &data1.u32);
      elementOffset += cuArray->desc.x / 8;
      if (cuArray->desc.y) {
        mem->read(tex_array_index + elementOffset, cuArray->desc.y / 8,
                  &data2.u32);
        elementOffset += cuArray->desc.y / 8;
        if (cuArray->desc.z) {
          mem->read(tex_array_index + elementOffset, cuArray->desc.z / 8,
                    &data3.u32);
          elementOffset += cuArray->desc.z / 8;
          if (cuArray->desc.w)
            mem->read(tex_array_index + elementOffset, cuArray->desc.w / 8,
                      &data4.u32);
        }
      }
      break;
    }
    case B64_TYPE:
    case U64_TYPE:
    case S64_TYPE:
      mem->read(tex_array_index, 8, &data1.u64);
      if (cuArray->desc.y) {
        mem->read(tex_array_index + 8, 8, &data2.u64);
        if (cuArray->desc.z) {
          mem->read(tex_array_index + 16, 8, &data3.u64);
          if (cuArray->desc.w) mem->read(tex_array_index + 24, 8, &data4.u64);
        }
      }
      break;
    case F16_TYPE:
      assert(0);
      break;
    case F32_TYPE: {
      if (texref->filterMode == cudaFilterModeLinear) {
        texAddr_t b_lim = wrap;
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          b_lim = clamp;
        }
        size_t elem_size = (cuArray->desc.x + cuArray->desc.y +
                            cuArray->desc.z + cuArray->desc.w) /
                           8;
        size_t elem_ofst = 0;

        data1.f32 =
            tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                              height, elem_size, alpha, beta, b_lim);
        elem_ofst += cuArray->desc.x / 8;
        if (cuArray->desc.y) {
          data2.f32 =
              tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                                height, elem_size, alpha, beta, b_lim);
          elem_ofst += cuArray->desc.y / 8;
          if (cuArray->desc.z) {
            data3.f32 =
                tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                                  height, elem_size, alpha, beta, b_lim);
            elem_ofst += cuArray->desc.z / 8;
            if (cuArray->desc.w)
              data4.f32 = tex_linf_sampling(mem, tex_array_base, x + elem_ofst,
                                            y, width, height, elem_size, alpha,
                                            beta, b_lim);
          }
        }
      } else {
        mem->read(tex_array_index, cuArray->desc.x / 8, &data1.f32);
        if (cuArray->desc.y) {
          mem->read(tex_array_index + 4, cuArray->desc.y / 8, &data2.f32);
          if (cuArray->desc.z) {
            mem->read(tex_array_index + 8, cuArray->desc.z / 8, &data3.f32);
            if (cuArray->desc.w)
              mem->read(tex_array_index + 12, cuArray->desc.w / 8, &data4.f32);
          }
        }
      }
    } break;
    case F64_TYPE:
    case FF64_TYPE:
      mem->read(tex_array_index, 8, &data1.f64);
      if (cuArray->desc.y) {
        mem->read(tex_array_index + 8, 8, &data2.f64);
        if (cuArray->desc.z) {
          mem->read(tex_array_index + 16, 8, &data3.f64);
          if (cuArray->desc.w) mem->read(tex_array_index + 24, 8, &data4.f64);
        }
      }
      break;
    default:
      assert(0);
      break;
  }
  int x_block_coord, y_block_coord, memreqindex, blockoffset;

  switch (dimension) {
    case GEOM_MODIFIER_1D:
      thread->m_last_effective_address = tex_array_index;
      break;
    case GEOM_MODIFIER_2D:
      x_block_coord = x >> (texInfo->Tx_numbits + texInfo->texel_size_numbits);
      y_block_coord = y >> texInfo->Ty_numbits;

      memreqindex =
          ((y_block_coord * cuArray->width / texInfo->Tx) + x_block_coord) << 6;

      blockoffset = (x % (texInfo->Tx * texInfo->texel_size) +
                     (y % (texInfo->Ty)
                      << (texInfo->Tx_numbits + texInfo->texel_size_numbits)));
      memreqindex += blockoffset;
      thread->m_last_effective_address =
          tex_array_base + memreqindex;  // tex_array_index;
      break;
    default:
      assert(0);
  }
  thread->m_last_memory_space = tex_space;

  // normalize output into floating point numbers according to the texture read
  // mode
  if (texAttr->m_readmode == cudaReadModeNormalizedFloat) {
    textureNormalizeOutput(cuArray->desc, data1, data2, data3, data4);
  } else {
    assert(texAttr->m_readmode == cudaReadModeElementType);
  }

  thread->set_vector_operand_values(dst, data1, data2, data3, data4);
#endif
}

void txq_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void trap_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vabsdiff_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vadd_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vmad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

#define VMAX 0
#define VMIN 1

void vmax_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  video_mem_instruction(pI, thread, VMAX);
}
void vmin_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  video_mem_instruction(pI, thread, VMIN);
}
void vset_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vshl_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vshr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vsub_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void vote_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static bool first_in_warp = true;
  static bool and_all;
  static bool or_all;
  static unsigned int ballot_result;
  static std::list<ptx_thread_info *> threads_in_warp;
  static unsigned last_tid;

  if (first_in_warp) {
    first_in_warp = false;
    threads_in_warp.clear();
    and_all = true;
    or_all = false;
    ballot_result = 0;
    int offset = 31;
    while ((offset >= 0) && !pI->active(offset)) offset--;
    assert(offset >= 0);
    last_tid =
        (thread->get_hw_tid() - (thread->get_hw_tid() % pI->warp_size())) +
        offset;
  }

  ptx_reg_t src1_data;
  const operand_info &src1 = pI->src1();
  src1_data = thread->get_operand_value(src1, pI->dst(), PRED_TYPE, thread, 1);

  // predicate value was changed so the lowest bit being set means the zero flag
  // is set. As a result, the value of src1_data.pred must be inverted to get
  // proper behavior
  bool pred_value = !(src1_data.pred & 0x0001);
  bool invert = src1.is_neg_pred();

  threads_in_warp.push_back(thread);
  and_all &= (invert ^ pred_value);
  or_all |= (invert ^ pred_value);

  // vote.ballot
  if (invert ^ pred_value) {
    int lane_id = thread->get_hw_tid() % pI->warp_size();
    ballot_result |= (1 << lane_id);
  }

  if (thread->get_hw_tid() == last_tid) {
    if (pI->vote_mode() == ptx_instruction::vote_ballot) {
      ptx_reg_t data = ballot_result;
      for (std::list<ptx_thread_info *>::iterator t = threads_in_warp.begin();
           t != threads_in_warp.end(); ++t) {
        const operand_info &dst = pI->dst();
        (*t)->set_operand_value(dst, data, pI->get_type(), (*t), pI);
      }
    } else {
      bool pred_value = false;

      switch (pI->vote_mode()) {
        case ptx_instruction::vote_any:
          pred_value = or_all;
          break;
        case ptx_instruction::vote_all:
          pred_value = and_all;
          break;
        case ptx_instruction::vote_uni:
          pred_value = (or_all ^ and_all);
          break;
        default:
          abort();
      }
      ptx_reg_t data;
      data.pred = pred_value ? 0 : 1;  // the way ptxplus handles the zero flag,
                                       // 1 = false and 0 = true

      for (std::list<ptx_thread_info *>::iterator t = threads_in_warp.begin();
           t != threads_in_warp.end(); ++t) {
        const operand_info &dst = pI->dst();
        (*t)->set_operand_value(dst, data, PRED_TYPE, (*t), pI);
      }
    }
    first_in_warp = true;
  }
}

void activemask_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  active_mask_t l_activemask_bitset = pI->get_warp_active_mask();
  uint32_t l_activemask_uint =
      static_cast<uint32_t>(l_activemask_bitset.to_ulong());

  const operand_info &dst = pI->dst();
  thread->set_operand_value(dst, l_activemask_uint, U32_TYPE, thread, pI);
}

void xor_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(~(src1_data.pred) ^ ~(src2_data.pred));
  else
    data.u64 = src1_data.u64 ^ src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void inst_not_implemented(const ptx_instruction *pI) {
  printf(
      "GPGPU-Sim PTX: ERROR (%s:%u) instruction \"%s\" not (yet) implemented\n",
      pI->source_file(), pI->source_line(), pI->get_opcode_cstr());
  abort();
}

ptx_reg_t srcOperandModifiers(ptx_reg_t opData, operand_info opInfo,
                              operand_info dstInfo, unsigned type,
                              ptx_thread_info *thread) {
  ptx_reg_t result;
  memory_space *mem = NULL;
  size_t size;
  int t;
  result.u64 = 0;

  // complete other cases for reading from memory, such as reading from other
  // const memory
  if (opInfo.get_addr_space() == global_space) {
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);
    mem->read(opData.u32, size / 8, &result.u64);
    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);
  } else if (opInfo.get_addr_space() == shared_space) {
    mem = thread->m_shared_mem;
    type_info_key::type_decode(type, size, t);
    mem->read(opData.u32, size / 8, &result.u64);

    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);

  } else if (opInfo.get_addr_space() == const_space) {
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);

    mem->read((opData.u32 + opInfo.get_const_mem_offset()), size / 8,
              &result.u64);

    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);
  } else {
    result = opData;
  }

  if (opInfo.get_operand_lohi() == 1) {
    result.u64 = result.u64 & 0xFFFF;
  } else if (opInfo.get_operand_lohi() == 2) {
    result.u64 = (result.u64 >> 16) & 0xFFFF;
  }

  if (opInfo.get_operand_neg() == true) {
    result.f32 = -result.f32;
  }

  return result;
}

void video_mem_instruction(const ptx_instruction *pI, ptx_thread_info *thread,
                           int op_code) {
  const operand_info &dst = pI->dst();    // d
  const operand_info &src1 = pI->src1();  // a
  const operand_info &src2 = pI->src2();  // b
  const operand_info &src3 = pI->src3();  // c

  const unsigned i_type = pI->get_type();

  std::list<int> scalar_type;
  std::list<int> options;

  ptx_reg_t a, b, ta, tb, c, data;

  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  // TODO: implement this
  // ta = partSelectSignExtend( a, atype );
  // tb = partSelectSignExtend( b, btype );
  ta = a;
  tb = b;

  options = pI->get_options();
  assert(options.size() == 1);

  auto option = options.begin();
  assert(*option == ATOMIC_MAX || *option == ATOMIC_MIN);

  switch (i_type) {
    case S32_TYPE: {
      // assert all operands are S32_TYPE:
      scalar_type = pI->get_scalar_type();
      for (std::list<int>::iterator scalar = scalar_type.begin();
           scalar != scalar_type.end(); scalar++) {
        assert(*scalar == S32_TYPE);
      }
      assert(scalar_type.size() == 3);
      scalar_type.clear();

      switch (op_code) {
        case VMAX:
          data.s32 = MY_MAX_I(ta.s32, tb.s32);
          break;
        case VMIN:
          data.s32 = MY_MIN_I(ta.s32, tb.s32);
          break;
        default:
          assert(0);
      }

      switch (*option) {
        case ATOMIC_MAX:
          data.s32 = MY_MAX_I(data.s32, c.s32);
          break;
        case ATOMIC_MIN:
          data.s32 = MY_MIN_I(data.s32, c.s32);
          break;
        default:
          assert(0);  // not yet implemented
      }
      break;
    }
    default:
      assert(0);  // not yet implemented
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);

  return;
}
