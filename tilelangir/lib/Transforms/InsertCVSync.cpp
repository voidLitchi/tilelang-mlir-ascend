// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/InsertCVSync.cpp
 * \brief TileLangIR insert CV synchronization flags pass.
 *
 */

#include <iostream>
#include <optional>

#include "tilelangir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINSERTCVSYNC
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-insert-cv-sync"

template <typename T> struct ArithConstantTypeTraits {
  static IntegerType get(MLIRContext *ctx) {
    return IntegerType::get(ctx, sizeof(T) * 8);
  }
};

static Value convertToI64(OpBuilder &builder, Location loc, Value val) {
  auto type = val.getType();
  if (type.isInteger(64))
    return val;
  if (type.isIndex())
    return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), val);
  if (type.isInteger())
    return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), val);
  llvm_unreachable("Unsupported type for conversion to i64");
}

static hivm::PIPE getCopyPipe(hivm::AddressSpace src, hivm::AddressSpace dst) {
  using Addr = hivm::AddressSpace;
  using Pipe = hivm::PIPE;

  // GM → L1 / UB : PIPE_MTE2
  if (src == Addr::GM && (dst == Addr::L1 || dst == Addr::UB))
    return Pipe::PIPE_MTE2;

  // L1 → GM : PIPE_MTE3
  // L1 → L0A / L0B / UB : PIPE_MTE1
  if (src == Addr::L1) {
    if (dst == Addr::GM)
      return Pipe::PIPE_MTE3;
    if (dst == Addr::L0A || dst == Addr::L0B || dst == Addr::UB)
      return Pipe::PIPE_MTE1;
  }

  // L0C → GM / L1 / UB : PIPE_FIX
  if (src == Addr::L0C &&
      (dst == Addr::GM || dst == Addr::L1 || dst == Addr::UB))
    return Pipe::PIPE_FIX;

  // UB → GM / L1 : PIPE_MTE3
  if (src == Addr::UB && (dst == Addr::GM || dst == Addr::L1))
    return Pipe::PIPE_MTE3;

  // 非法组合
  return Pipe::PIPE_UNASSIGNED;
}

template <typename T> class ArithConstantManager {
  static_assert(std::is_integral_v<T>, "T must be an integral type");

public:
  // --------------------------------------------------------------------------
  // 构造函数：扫描 region 头部连续的 arith.const，缓存类型匹配的常量，
  // 并将 OpBuilder 插入点设置在第一个非常量 op 之前。
  // --------------------------------------------------------------------------
  explicit ArithConstantManager(Region &region) {
    MLIRContext *ctx = region.getContext();
    // 使用 traits 确定目标类型（不再自动推断）
    constType = ArithConstantTypeTraits<T>::get(ctx);

    auto &block = region.front();
    auto it = block.begin();

    // 扫描连续的 arith.const，直到遇到第一个非常量 op
    while (it != block.end()) {
      auto constOp = dyn_cast<arith::ConstantOp>(*it);
      if (!constOp)
        break; // 非常量，停止扫描

      // 只处理整数常量
      if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
        // 仅当类型与目标类型完全一致（包括符号性）才缓存
        if (intAttr.getType() == constType) {
          T val = extractInt(intAttr.getValue());
          ConstCache.try_emplace(val, constOp.getResult());
        }
        // 类型不匹配的整数常量直接跳过（例如 i64 vs i32）
      }
      // 非整数常量（浮点、index、复杂类型等）也跳过
      ++it;
    }

    // 设置 builder 插入点为第一个非常量 op 之前（若全是常量则设在 block 末尾）
    arithConstantBuilder = std::make_shared<OpBuilder>(ctx);
    arithConstantBuilder->setInsertionPoint(&block, it);
  }

  // --------------------------------------------------------------------------
  // 获取值为 val 的常量。若已存在则直接返回，否则创建新常量并插入、缓存。
  // --------------------------------------------------------------------------
  Value getArithConstant(T val, Location loc) {
    auto it = ConstCache.find(val);
    if (it != ConstCache.end())
      return it->second;

    // 创建新常量
    unsigned bits = constType.getIntOrFloatBitWidth();
    auto intAttr =
        IntegerAttr::get(constType, APInt(bits, val, std::is_signed_v<T>));
    auto newConst = arithConstantBuilder->create<arith::ConstantOp>(
        loc, constType, intAttr);
    Value result = newConst.getResult();
    ConstCache[val] = result;
    return result;
  }

private:
  llvm::DenseMap<T, Value> ConstCache;
  std::shared_ptr<OpBuilder> arithConstantBuilder;
  Type constType;

  // 从 APInt 提取 T 类型的值，依据 T 的符号性解释
  static T extractInt(const APInt &apVal) {
    if constexpr (std::is_signed_v<T>) {
      return static_cast<T>(apVal.getSExtValue());
    } else {
      return static_cast<T>(apVal.getZExtValue());
    }
  }
};

/// 临时 Block 辅助类：
/// - 构造时创建一个新 Block 并挂在给定 Region 末尾。
/// - 析构时自动销毁临时 Block（及其内部未被移走的 Op）。
/// - 提供便捷的 create() 方法向临时 Block 末尾追加 Op。
/// - 提供 moveAllOpsBefore/moveAllOpsAfter 等整体搬移方法。
class TemporaryBlock {
public:
  /// 构造：创建一个空 Block，插入到 region 末尾。
  explicit TemporaryBlock(mlir::Region &region)
      : region(region), block(new mlir::Block()) {
    region.push_back(block);
    builder = std::make_unique<mlir::OpBuilder>(block, block->end());
  }

  /// 析构：销毁临时 Block 及其中剩余的所有 Op。
  ~TemporaryBlock() {
    if (block) {
      block->erase(); // 即使 Block 为空也可安全调用
    }
  }

  // 禁止拷贝和移动
  TemporaryBlock(const TemporaryBlock &) = delete;
  TemporaryBlock &operator=(const TemporaryBlock &) = delete;
  TemporaryBlock(TemporaryBlock &&) = delete;
  TemporaryBlock &operator=(TemporaryBlock &&) = delete;

  /// 获取底层的 mlir::Block 指针。
  mlir::Block *getBlock() const { return block; }

  /// 获取内部持有的 OpBuilder（方便设置插入点到任意位置）。
  mlir::OpBuilder &getBuilder() { return *builder; }

  /// 在临时 Block 的末尾创建一个 Op（等价于 push_back）。
  /// 用法：auto op = temp.create<mlir::scf::ForOp>(loc, lb, ub, step);
  template <typename OpTy, typename... Args>
  OpTy create(mlir::Location loc, Args &&...args) {
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

  // ---- 整体搬移方法 ----

  /// 将临时 Block 中的所有 Op 一次性移到 refOp 之前。
  void moveAllOpsBefore(mlir::Operation *refOp) {
    assert(block && "TemporaryBlock already released");
    mlir::Block *targetBlock = refOp->getBlock();
    auto it = mlir::Block::iterator(refOp);
    targetBlock->getOperations().splice(it, block->getOperations());
  }

  /// 将临时 Block 中的所有 Op 一次性移到 refOp 之后。
  void moveAllOpsAfter(mlir::Operation *refOp) {
    assert(block && "TemporaryBlock already released");
    mlir::Block *targetBlock = refOp->getBlock();
    auto it = mlir::Block::iterator(refOp);
    targetBlock->getOperations().splice(std::next(it), block->getOperations());
  }

  /// 将临时 Block 中的所有 Op 一次性移到 targetBlock 的开头。
  void moveAllOpsToStart(mlir::Block *targetBlock) {
    targetBlock->getOperations().splice(targetBlock->begin(),
                                        block->getOperations());
  }

  /// 将临时 Block 中的所有 Op 一次性移到 targetBlock 的末尾。
  void moveAllOpsToEnd(mlir::Block *targetBlock) {
    targetBlock->getOperations().splice(targetBlock->end(),
                                        block->getOperations());
  }

private:
  mlir::Region &region;
  mlir::Block *block;
  std::shared_ptr<mlir::OpBuilder> builder;
};

class SyncManager {
public:
  SyncManager() = delete;

  // 带参构造函数：接收外层func，设置builder于开头，创建临时scope于结尾
  explicit SyncManager(func::FuncOp &funcOp)
      : arithConstantManagerI32(funcOp.getBody()),
        arithConstantManagerI64(funcOp.getBody()),
        initTempBlock(funcOp.getBody()), clearTempBlock(funcOp.getBody()) {}

  void InsertBlockSyncPair(Operation *setOp, Value setTaskId, Operation *waitOp,
                           Value waitTaskId, size_t num_multi,
                           bool is_back_pressure = false);
  void InsertInitFlagsBefore(Operation *op) {
    initTempBlock.moveAllOpsBefore(op);
  }
  void InsertClearFlagsAfter(Operation *op) {
    clearTempBlock.moveAllOpsAfter(op);
  }

private:
  static constexpr size_t FLAG_ID_LIMIT = 16;
  size_t baseIdC2V = 0;
  size_t baseIdV2C = 0;
  ArithConstantManager<int32_t> arithConstantManagerI32;
  ArithConstantManager<int64_t> arithConstantManagerI64;
  TemporaryBlock initTempBlock;
  TemporaryBlock clearTempBlock;

  std::pair<hivm::TCoreType, hivm::PIPE> ExtractDMAInfo(Operation *op);

  void InsertBlockSet(OpBuilder &builder, Location loc,
                      const hivm::TCoreType &thisCore,
                      const hivm::PIPE &thisPipe, Value taskId, Value multi,
                      Value flagOffset);
  void InsertBlockWait(OpBuilder &builder, Location loc,
                       const hivm::TCoreType &thisCore,
                       const hivm::PIPE &thisPipe, Value taskId, Value multi,
                       Value flagOffset);
};

void mlir::tilelangir::SyncManager::InsertBlockSyncPair(
    Operation *setOp, Value setTaskId, Operation *waitOp, Value waitTaskId,
    size_t num_multi, bool is_back_pressure) {
  assert(arithConstantBuilder &&
         "SyncManager created by default could not be used.");
  auto [setCoreType, setPipe] = ExtractDMAInfo(setOp);
  auto [waitCoreType, waitPipe] = ExtractDMAInfo(waitOp);
  OpBuilder builder(setOp->getContext());
  auto setLoc = setOp->getLoc();
  auto waitLoc = waitOp->getLoc();

  Value multi = arithConstantManagerI64.getArithConstant(
      static_cast<int64_t>(num_multi), setLoc);

  Value flagOffset;
  if (setCoreType == hivm::TCoreType::CUBE) {
    flagOffset = arithConstantManagerI64.getArithConstant(
        static_cast<int64_t>(baseIdC2V), setLoc);
    baseIdC2V += num_multi;
  } else if (setCoreType == hivm::TCoreType::VECTOR) {
    flagOffset = arithConstantManagerI64.getArithConstant(
        static_cast<int64_t>(baseIdV2C), setLoc);
    baseIdV2C += num_multi;
  } else {
    assert(false && "Invalid core type detected.");
  }

  builder.setInsertionPointAfter(setOp);
  InsertBlockSet(builder, setLoc, setCoreType, setPipe,
                 convertToI64(builder, setLoc, setTaskId), multi, flagOffset);
  builder.setInsertionPoint(waitOp);
  InsertBlockWait(builder, waitLoc, waitCoreType, waitPipe,
                  convertToI64(builder, waitLoc, waitTaskId), multi,
                  flagOffset);

  if (is_back_pressure) {
    Value valueZero = arithConstantManagerI32.getArithConstant(0, setLoc);
    Value valueOne = arithConstantManagerI32.getArithConstant(1, setLoc);
    Value multiI32 = arithConstantManagerI32.getArithConstant(
        static_cast<int32_t>(num_multi), setLoc);

    auto initForOp =
        initTempBlock.create<scf::ForOp>(setLoc, valueZero, multiI32, valueOne);
    auto initCoreTypeAttr =
        mlir::hivm::TCoreTypeAttr::get(builder.getContext(), setCoreType);
    initForOp->setAttr(hivm::TCoreTypeAttr::name, initCoreTypeAttr);
    {
      builder.setInsertionPointToStart(initForOp.getBody());
      Value initId = initForOp.getInductionVar();
      initId = convertToI64(builder, initForOp->getLoc(), initId);
      InsertBlockSet(builder, initForOp->getLoc(), setCoreType, setPipe, initId,
                     multi, flagOffset);
    }

    auto clearForOp = clearTempBlock.create<scf::ForOp>(waitLoc, valueZero,
                                                        multiI32, valueOne);
    auto clearCoreTypeAttr =
        mlir::hivm::TCoreTypeAttr::get(builder.getContext(), waitCoreType);
    clearForOp->setAttr(hivm::TCoreTypeAttr::name, clearCoreTypeAttr);
    {
      builder.setInsertionPointToStart(clearForOp.getBody());
      Value clearId = clearForOp.getInductionVar();
      clearId = convertToI64(builder, clearForOp->getLoc(), clearId);
      InsertBlockWait(builder, clearForOp->getLoc(), waitCoreType, waitPipe,
                      clearId, multi, flagOffset);
    }
  }
}

void mlir::tilelangir::SyncManager::InsertBlockSet(
    OpBuilder &builder, Location loc, const hivm::TCoreType &thisCore,
    const hivm::PIPE &thisPipe, Value taskId, Value multi, Value flagOffset) {
  Value id = builder.create<arith::RemSIOp>(loc, taskId, multi);
  Value flagIdWithOffset = builder.create<arith::AddIOp>(loc, id, flagOffset);
  Value flagIdLimit = arithConstantManagerI64.getArithConstant(
      static_cast<int64_t>(FLAG_ID_LIMIT), loc);
  Value finalFlagId =
      builder.create<arith::RemSIOp>(loc, flagIdWithOffset, flagIdLimit);

  auto coreSrcAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), thisCore);
  auto sync_mode = hivm::SyncBlockInstrModeAttr::get(
      builder.getContext(),
      hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
  auto tPipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto pipTypeAttr = hivm::PipeAttr::get(builder.getContext(), thisPipe);

  builder.create<hivm::SyncBlockSetOp>(loc, coreSrcAttr, pipTypeAttr,
                                       tPipTypeAttr, finalFlagId, mlir::Value(),
                                       sync_mode);
}

void mlir::tilelangir::SyncManager::InsertBlockWait(
    OpBuilder &builder, Location loc, const hivm::TCoreType &thisCore,
    const hivm::PIPE &thisPipe, Value taskId, Value multi, Value flagOffset) {
  Value id = builder.create<arith::RemSIOp>(loc, taskId, multi);
  Value flagIdWithOffset = builder.create<arith::AddIOp>(loc, id, flagOffset);
  Value flagIdLimit = arithConstantManagerI64.getArithConstant(
      static_cast<int64_t>(FLAG_ID_LIMIT), loc);
  Value finalFlagId =
      builder.create<arith::RemSIOp>(loc, flagIdWithOffset, flagIdLimit);

  auto coreSrcAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), thisCore);
  auto tPipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto pipTypeAttr = hivm::PipeAttr::get(builder.getContext(), thisPipe);

  builder.create<hivm::SyncBlockWaitOp>(loc, coreSrcAttr, tPipTypeAttr,
                                        pipTypeAttr, finalFlagId);
}

std::pair<hivm::TCoreType, hivm::PIPE>
mlir::tilelangir::SyncManager::ExtractDMAInfo(Operation *op) {
  using Addr = hivm::AddressSpace;

  // 1. 提取地址空间的辅助 lambda（assert具有地址空间信息）
  auto getAddressSpace = [](Value val) -> Addr {
    auto memrefType = val.getType().cast<MemRefType>();
    auto attr =
        memrefType.getMemorySpace().dyn_cast_or_null<hivm::AddressSpaceAttr>();
    assert(attr && "InsertCVSync Pass rely on scope information.");
    return attr ? attr.getAddressSpace() : Addr::Zero;
  };

  // 2. 尝试转换为 memref.copy/hivm.hir.nd2nz/hivm.hir.fixpipe
  Addr src = Addr::GM, dst = Addr::GM;

  if (auto copyOp = llvm::dyn_cast<memref::CopyOp>(op)) {
    src = getAddressSpace(copyOp.getSource());
    dst = getAddressSpace(copyOp.getTarget());
  } else if (auto copyOp = llvm::dyn_cast<hivm::FixpipeOp>(op)) {
    src = getAddressSpace(copyOp.getSrc());
    dst = getAddressSpace(copyOp.getDst());
  } else if (auto copyOp = llvm::dyn_cast<hivm::ND2NZOp>(op)) {
    src = getAddressSpace(copyOp.getSrc());
    dst = getAddressSpace(copyOp.getDst());
  } else {
    assert(false && "Fail to recognize as DMA Op");
  }

  // 3. 计算 PIPE（复用独立函数）
  hivm::PIPE pipe = getCopyPipe(src, dst);

  // 4. 计算 TCoreType（忽略 GM）
  llvm::SmallSet<Addr, 4> spaces;
  if (src != Addr::GM)
    spaces.insert(src);
  if (dst != Addr::GM)
    spaces.insert(dst);

  hivm::TCoreType core;
  if (spaces.empty()) {
    core = hivm::TCoreType::CUBE_OR_VECTOR;
  } else {
    bool hasUB = spaces.count(Addr::UB);
    bool hasL = spaces.count(Addr::L1) || spaces.count(Addr::L0A) ||
                spaces.count(Addr::L0B) || spaces.count(Addr::L0C);
    if (hasUB && hasL)
      core = hivm::TCoreType::CUBE_AND_VECTOR;
    else if (hasUB && !hasL)
      core = hivm::TCoreType::VECTOR;
    else if (!hasUB && hasL)
      core = hivm::TCoreType::CUBE;
    else
      core = hivm::TCoreType::CUBE_OR_VECTOR; // 纯 GM 不会进这里
  }

  return {core, pipe};
}

struct TileLangIRInsertCVSync
    : impl::TileLangIRInsertCVSyncBase<TileLangIRInsertCVSync> {
public:
  void runOnOperation() override;

private:
  std::shared_ptr<SyncManager> syncManager;

  void processAllocWorkspace(bishengir::memref_ext::AllocWorkspaceOp allocOp,
                             Region &region);
  void InsertCVSyncInSinglePipeline(Operation *op);
};

void TileLangIRInsertCVSync::processAllocWorkspace(
    bishengir::memref_ext::AllocWorkspaceOp allocOp, mlir::Region &region) {
  Value workspace = allocOp.getResult();

  // ── 1. 读取 multi_buffer 属性 ─────────────────────────────────
  llvm::outs() << "processAllocWorkspace: Stage 1\n";
  int64_t multiBufferVal = 1;
  for (Operation *user : workspace.getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      if (auto attr = markOp->template getAttrOfType<mlir::IntegerAttr>(
              "hivm.multi_buffer")) {
        multiBufferVal = attr.getInt();
      }
      break;
    }
    if (user->getName().getStringRef() == "annotation.mark") {
      if (auto attr =
              user->getAttrOfType<mlir::IntegerAttr>("hivm.multi_buffer")) {
        multiBufferVal = attr.getInt();
      }
      break;
    }
  }

  // ── 2. 收集 region 中所有一级 tcore_type scope 并过滤出使用 workspace 的 ──
  llvm::outs() << "processAllocWorkspace: Stage 2\n";
  SmallVector<Operation *> orderedScopes; // 有序的、真正使用 workspace 的 scope
  SmallVector<hivm::TCoreType> coreTypes;
  SmallVector<Value> taskIds;

  // 获取 pipeline 算法属性（来自 region 的父操作）
  Operation *commonParent = region.getParentOp();
  std::optional<StringRef> pipelineAlg;
  if (auto attr = commonParent->getAttrOfType<mlir::StringAttr>(
          "tilelangir.pipeline_alg"))
    pipelineAlg = attr.getValue();
  assert(pipelineAlg && "Fail to get the correct pipeline algorithm.");
  llvm::outs() << "processAllocWorkspace: Stage 2 - alg: "
               << pipelineAlg.value() << "\n";

  // 遍历 region 的第一个 basic block 中的直接子操作
  if (region.empty())
    return;
  Block &block = region.front();
  for (Operation &op : block) {
    // 调试：打印每个一级操作的名字
    llvm::outs() << "Top-level op: " << op.getName().getStringRef() << "\n";
    // 打印该操作的所有属性名
    for (auto namedAttr : op.getAttrs()) {
      llvm::outs() << "  has attr: " << namedAttr.getName().strref() << "\n";
    }

    // 检查该操作自身是否有 hivm.tcore_type 属性且为 VECTOR/CUBE
    auto attr =
        op.getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!attr) {
      llvm::outs() << "processAllocWorkspace: Stage 2 - No Core Type\n";
      continue;
    }
    auto ct = attr.getTcoretype();
    if (ct != hivm::TCoreType::VECTOR && ct != hivm::TCoreType::CUBE) {
      llvm::outs() << "processAllocWorkspace: Stage 2 - Invalid Core Type\n";
      continue;
    }

    // 暂存，等 reachableValues 计算后再判断是否使用 workspace
    orderedScopes.push_back(&op);
    coreTypes.push_back(ct);

    // 若 pipeline 为 unroll，该 scope 必须是 scf.for，记录其归纳变量
    if (pipelineAlg.value() == "unroll") {
      auto forOp = dyn_cast<scf::ForOp>(&op);
      assert(forOp &&
             "For pipeline implemented by unroll, for loop must be remained");
      taskIds.push_back(forOp.getInductionVar());
    } else {
      // 非 unroll 模式下 taskIds 不需要填充，但为了索引一致，推入空 Value
      taskIds.push_back(Value());
    }
  }

  if (orderedScopes.size() <= 1) {
    llvm::outs()
        << "processAllocWorkspace: Stage 2 - \"orderedScopes.size() <= 1\"\n";
    return; // 只有一个 scope，不会发生跨核
  }

  // ── 3. 预计算 workspace 的所有视图派生值集合 (reachableValues) ──
  llvm::outs() << "processAllocWorkspace: Stage 3\n";
  SmallPtrSet<Value, 8> reachableValues;
  SmallVector<Value> worklist;
  reachableValues.insert(workspace);
  worklist.push_back(workspace);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(user)) {
        for (Value res : user->getResults()) {
          if (reachableValues.insert(res).second)
            worklist.push_back(res);
        }
      }
    }
  }

  // ── 4. 过滤 orderedScopes，只保留内部包含 reachableValues 最终使用的 scope
  // ──
  llvm::outs() << "processAllocWorkspace: Stage 4\n";
  // 注意：需要同步过滤 coreTypes 和 taskIds
  SmallVector<Operation *> activeScopes;
  SmallVector<hivm::TCoreType> activeCoreTypes;
  SmallVector<Value> activeTaskIds;

  auto scopeUsesWorkspace = [&](Operation *scope) -> bool {
    if (scope->getNumRegions() == 0)
      return false;
    bool found = false;
    scope->getRegion(0).walk([&](Operation *nested) {
      if (isa<mlir::ViewLikeOpInterface>(nested))
        return WalkResult::advance(); // 跳过纯视图操作
      for (Value operand : nested->getOperands()) {
        if (reachableValues.contains(operand)) {
          found = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return found;
  };

  for (size_t i = 0; i < orderedScopes.size(); ++i) {
    if (scopeUsesWorkspace(orderedScopes[i])) {
      activeScopes.push_back(orderedScopes[i]);
      activeCoreTypes.push_back(coreTypes[i]);
      activeTaskIds.push_back(taskIds[i]);
    }
  }

  if (activeScopes.size() <= 1)
    return; // 使用 workspace 的 scope 不足两个，无跨核

  // ── 5. 检测跨核并插入同步 ─────────────────────────────────────
  llvm::outs() << "processAllocWorkspace: Stage 5\n";
  size_t n = activeCoreTypes.size();
  bool hasSync = false;

  for (size_t i = 0; i < n; ++i) {
    hivm::TCoreType cur = activeCoreTypes[i];
    hivm::TCoreType next = activeCoreTypes[(i + 1) % n];
    if (cur == next)
      continue;

    hasSync = true;
    Operation *scopeA = activeScopes[i];
    Operation *scopeB = activeScopes[(i + 1) % n];

    // 辅助函数：收集 scope body 中所有“包含最终使用者”的直接子操作
    auto getDirectChildOpsUsingValue =
        [&](Operation *scope) -> SmallVector<Operation *> {
      SmallVector<Operation *> childOps;
      if (scope->getNumRegions() == 0)
        return childOps;

      Block &body = scope->getRegion(0).front();
      for (Operation &op : body) {
        if (op.hasTrait<mlir::OpTrait::IsTerminator>())
          continue;

        bool hasRealUse = false;
        op.walk([&](Operation *nested) {
          if (isa<mlir::ViewLikeOpInterface>(nested))
            return WalkResult::advance();
          for (Value operand : nested->getOperands()) {
            if (reachableValues.contains(operand)) {
              hasRealUse = true;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
        if (hasRealUse)
          childOps.push_back(&op);
      }
      return childOps;
    };

    auto usersA = getDirectChildOpsUsingValue(scopeA);
    auto usersB = getDirectChildOpsUsingValue(scopeB);

    assert(!usersA.empty() && !usersB.empty());

    // 提取 taskId：若 pipeline 为 unroll 则取对应的归纳变量，否则为空 Value
    Value taskIdA = activeTaskIds[i];
    Value taskIdB = activeTaskIds[(i + 1) % n];

    syncManager->InsertBlockSyncPair(usersA.back(), taskIdA, usersB.back(),
                                     taskIdB, multiBufferVal, i == n - 1);
  }

  // 无跨核则静默返回
  (void)hasSync;
}

void TileLangIRInsertCVSync::runOnOperation() {
  // 先收集所有外层循环，避免在遍历中修改
  SmallVector<scf::ForOp> pipelineLoops;
  getOperation()->walk([&](scf::ForOp forOp) {
    if (forOp->getAttr("tilelangir.pipeline_alg"))
      pipelineLoops.push_back(forOp);
  });

  for (scf::ForOp pipelineLoop : pipelineLoops) {
    InsertCVSyncInSinglePipeline(pipelineLoop);
    // InsertInitAndClearInSinglePipeline(pipelineLoop);
  }
}

void TileLangIRInsertCVSync::InsertCVSyncInSinglePipeline(Operation *op) {
  Region *region = nullptr;
  if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
    region = &funcOp.getBody(); // getBody() 返回 Region&，取地址
  } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
    region = &forOp.getRegion(); // getRegion() 返回 Region&，取地址
  } else {
    assert(false && "Invalid op type for pipeline.");
  }

  auto module = getOperation();

  // 统计模块中的函数个数
  auto funcOps = module.getBody()->getOps<mlir::func::FuncOp>();
  if (std::distance(funcOps.begin(), funcOps.end()) != 1)
    return; // 多于一个函数时直接跳过

  // 初始化flagId管理器
  auto func = *funcOps.begin();
  llvm::outs() << "Initializing SyncManager\n";
  syncManager = std::make_shared<SyncManager>(func);

  getOperation()->walk([&](bishengir::memref_ext::AllocWorkspaceOp op) {
    llvm::outs() << "Processing AllocWorkspaceOp at loc: " << op.getLoc()
                 << "\n";
    processAllocWorkspace(op, *region);
  });
  syncManager->InsertInitFlagsBefore(op);
  syncManager->InsertClearFlagsAfter(op);
  syncManager.reset();
}

#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
