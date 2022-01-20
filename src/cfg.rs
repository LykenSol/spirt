//! Control-flow graph (CFG) abstractions and utilities.

use crate::{spv, AttrSet, EntityKeyedDenseMap, FxIndexMap, Region, Value};
use smallvec::SmallVec;

/// The control-flow graph (CFG) of a function, as control-flow instructions
/// (`ControlInst`s) attached to region-relative CFG points (`ControlPoint`s).
#[derive(Default)]
pub struct ControlFlowGraph {
    /// Effectively a `Map<ControlPoint, ControlInst>`, with efficient storage.
    pub control_insts: EntityKeyedDenseMap<Region, PerEntryAndExit<Option<ControlInst>>>,
}

impl ControlFlowGraph {
    pub fn control_inst_at(&self, point: ControlPoint) -> Option<&ControlInst> {
        let both = self.control_insts.get(point.region())?;
        match point {
            ControlPoint::Entry(_) => both.entry.as_ref(),
            ControlPoint::Exit(_) => both.exit.as_ref(),
        }
    }

    pub fn control_inst_mut_at(&mut self, point: ControlPoint) -> Option<&mut ControlInst> {
        let both = self.control_insts.get_mut(point.region())?;
        match point {
            ControlPoint::Entry(_) => both.entry.as_mut(),
            ControlPoint::Exit(_) => both.exit.as_mut(),
        }
    }

    /// Iterate over all `ControlPoint`s reachable through the CFG, starting at
    /// `entry`, in reverse post-order (RPO).
    ///
    /// RPO iteration over a CFG provides certain guarantees, most importantly
    /// that SSA definitions are visited before any of their uses.
    pub fn rev_post_order(
        &self,
        entry: ControlPoint,
    ) -> impl DoubleEndedIterator<Item = ControlPoint> {
        self.post_order(entry).rev()
    }

    /// Iterate over all `ControlPoint`s reachable through the CFG, starting at
    /// `entry`, in post-order.
    pub fn post_order(&self, entry: ControlPoint) -> impl DoubleEndedIterator<Item = ControlPoint> {
        let mut post_order = SmallVec::<[_; 8]>::new();
        {
            let mut visited = PerEntryAndExit {
                entry: EntityKeyedDenseMap::new(),
                exit: EntityKeyedDenseMap::new(),
            };
            self.post_order_step(entry, &mut visited, &mut post_order);
        }

        post_order.into_iter()
    }

    fn post_order_step(
        &self,
        point: ControlPoint,
        // FIXME(eddyb) use dense entity-keyed bitsets here instead.
        visited: &mut PerEntryAndExit<EntityKeyedDenseMap<Region, ()>>,
        post_order: &mut SmallVec<[ControlPoint; 8]>,
    ) {
        let already_visited = {
            let visited = match point {
                ControlPoint::Entry(_) => &mut visited.entry,
                ControlPoint::Exit(_) => &mut visited.exit,
            };
            visited.insert(point.region(), ()).is_some()
        };
        if already_visited {
            return;
        }

        if let Some(control_inst) = self.control_inst_at(point) {
            for &target in &control_inst.targets {
                self.post_order_step(target, visited, post_order);
            }
        } else {
            // Blocks don't have `ControlInst`s attached to their `Entry`,
            // only to their `Exit`, but we don't have access to the `RegionDef`
            // to confirm - however, only blocks should have this distinction.
            if let ControlPoint::Entry(region) = point {
                let target = ControlPoint::Exit(region);
                self.post_order_step(target, visited, post_order);
            }
        }
        post_order.push(point);
    }
}

/// A point in the control-flow graph (CFG) of a function, relative to a `Region`.
///
/// The whole CFG of the function consists of `ControlInst`s connecting all such
/// points, expect for these special cases:
///
/// * `RegionKind::UnstructuredMerge`: lacks an `Entry` point entirely, as its
///   purpose is to represent an effectively multiple-entry single-exit (MESE)
///   "half-region", that could only become a proper region by structurization
///   (and would likely end up the "merge" / exit side of a structured region)
///
/// * `RegionKind::Block`: between its `Entry` and `Exit` points, a block only
///   has its own linear sequence of instructions as (implied) control-flow, so
///   no `ControlInst` can attach to its `Entry` or target its `Exit`
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum ControlPoint {
    Entry(Region),
    Exit(Region),
}

impl ControlPoint {
    pub fn region(self) -> Region {
        match self {
            Self::Entry(r) | Self::Exit(r) => r,
        }
    }
}

pub struct PerEntryAndExit<T> {
    pub entry: T,
    pub exit: T,
}

pub struct ControlInst {
    pub attrs: AttrSet,

    pub kind: ControlInstKind,

    pub inputs: SmallVec<[Value; 2]>,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    pub targets: SmallVec<[ControlPoint; 4]>,

    /// The `Value` in `target_merge_outputs[region][output_idx]` is the one
    /// that `Value::RegionOutput { region, output_idx }` will take on exiting
    /// `region` (via a `ControlPoint::Exit(region)` in `targets`).
    pub target_merge_outputs: FxIndexMap<Region, SmallVec<[Value; 2]>>,
}

pub enum ControlInstKind {
    /// Reaching this point in the control-flow is undefined behavior, e.g.:
    /// * a `SelectBranch` case that's known to be impossible
    /// * after a function call, where the function never returns
    ///
    /// Optimizations can take advantage of this information, to assume that any
    /// necessary preconditions for reaching this point, are never met.
    Unreachable,

    /// Leave the current function, optionally returning a value.
    Return,

    /// Leave the current invocation, similar to returning from every function
    /// call in the stack (up to and including the entry-point), but potentially
    /// indicating a fatal error as well.
    ExitInvocation(ExitInvocationKind),

    /// Unconditional branch to a single target.
    Branch,

    /// Branch to one of several targets, chosen by a single value input.
    SelectBranch(SelectionKind),
}

pub enum ExitInvocationKind {
    SpvInst(spv::Inst),
}

pub enum SelectionKind {
    /// Conditional branch on boolean condition, i.e. `if`-`else`.
    BoolCond,

    SpvInst(spv::Inst),
}
