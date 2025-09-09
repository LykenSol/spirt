//! Control-flow abstractions and passes.
//
// FIXME(eddyb) consider moving more definitions into this module.

use crate::{scalar, spv};

// NOTE(eddyb) all the modules are declared here, but they're documented "inside"
// (i.e. using inner doc comments).
pub mod callgraph;
pub mod cfgssa;
pub mod structurize;
pub mod unstructured;

// FIXME(eddyb) consider interning this.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum SelectionKind {
    /// Two-case selection based on boolean condition, i.e. `if`-`else`, with
    /// the two cases being "then" and "else" (in that order).
    BoolCond,

    /// `N+1`-case selection based on comparing an integer scrutinee against
    /// `N` constants, i.e. `switch`, with the last case being the "default"
    /// (making it the only case without a matching entry in `case_consts`).
    Switch {
        // FIXME(eddyb) avoid some of the `scalar::Const` overhead here, as there
        // is only a single type and we shouldn't need to store more bits per case,
        // than the actual width of the integer type.
        // FIXME(eddyb) consider storing this more like sorted compressed keyset,
        // as there can be no duplicates, and in many cases it may be contiguous.
        case_consts: Vec<scalar::Const>,
    },
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ExitInvocationKind {
    SpvInst(spv::Inst),
    // NOTE(eddyb) okay so for the name, maybe something about "guarantee diverge"
    // or "guarantee exit" hmmmmmm. "delegate exit to caller?" - anyway the idea
    // to *stress* in docs that it allows inlining to avoid restructuring, because
    // the behavior can be replicated by *copying* the postdom `ExitInvocation`
    // from the caller, at every usage of this thing in the callee being inlined!
    // TODO(eddyb) add `ForceCallerExit`, which has to be propagated in the caller,
    // transitively, up to an entry-point (which can either use `ForceCallerExit`,
    // even tho that name isn't perfect... or abuse `SpvInst(OpReturn)`, hmpf),
    // and this is (theoretically) enforced as such:
    // - pick an index for the "force caller exit condition", which must be a boolean
    //   that will be set to `true` to exit, and all other return values will be
    //   undef (in the general non-`bool` case, `scalar::Const` could be used?)
    //   - by making the "force condition" *optional*, its lack would imply that
    //     the exit is *unconditional* in the caller, which is accurate for an
    //     entry-point (i.e. if you wrap the entry-point in a function, it would
    //     never need to branch on anything to exit, since there is nothing that
    //     needs to run *after* the entry-point, and this is a *successful* exit)
    //   - this would also prevent entry-points from being called from anywhere,
    //     i.e. forcing it to be specifically an entry-point
    // - the `Call` side needs to be pattern-matching on the outputs, specifically
    //   the "force caller exit condition", e.g. `(a, b) = call ...` followed by
    //   `if b { ExitInvocation(...) }`
    //   - it's tempting to force this into the `Call` node, but what if was just
    //     the successor node? because that would help with recursion emulation,
    //     since it would just, and even `spv::lift` would be p simple...
    //   - (in the general non-`bool` case, it would need `scalar::Const`s and
    //     be more like e.g. `switch b { 123 => ExitInvocation(...), ... }`,
    //     so it can be forced to e.g. `FxIndexMap<scalar::Const, ExitInvocation>`,
    //     which in theory could be used to (expensively) encode backtraces etc.)
    // - checking would consist of ensuring that all callers of a function using
    //   `ForceCallerExit` also have the right conditional `ExitInvocation` at
    //   each call site - notably, this has these nice properties:
    //   1. transitivity of `ExitInvocation` means that the propagation doesn't
    //      need to enforced on *chains* of calls, just very locally, and if
    //      there are multiple valid `ExitInvocationKind`s, *they are allowed*
    //      (i.e. it doesn't need to be `ForceCallerExit` up to the entry point)
    //   2. some part of the function signature (or an attribute) could make
    //      the checks *far more local*, like "forced caller exit condition"
    //      can be enforced at both `ForceCallerExit`s inside the definition,
    //      and then at every call site (i.e. all calls should be post-dominated
    //      by a check of that condition which leads to an `ExitInvocation`)
    //      - attribute might be bad because this is basically "part of the ABI"
    //        of the function, and should be part of function pointers as a kind
    //        of "may unwind" (should this feature talk about "unwinding"? the
    //        issue there is that this is not a "true" unwinder design, hmpf)
    //   3. returning the force-exit-in-caller values through regular return
    //      is *entirely* allowed, and could even serve some purpose!
}
