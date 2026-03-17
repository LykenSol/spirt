//! Scheduling of execution and resources (e.g. register allocation).
//
// NOTE(eddyb) while SPIR-T is still strictly ordered, it might not be in
// the future, and a RVSDG-like approach would require an explicit ordering.

use crate::func_at::FuncAt;
use crate::{EntityOrientedDenseMap, FxIndexSet, Node, NodeKind, Region, Value, Var};
use itertools::Itertools as _;
use std::collections::VecDeque;
use std::mem;
use std::num::NonZeroU32;

// FIXME(eddyb) try to separate {control,data}-flow?
#[derive(Default)]
pub struct Schedule {
    pub regions: EntityOrientedDenseMap<Region, RegionSchedule>,
    pub vars: EntityOrientedDenseMap<Var, VarSchedule>,
}

pub struct RegionSchedule {
    pub nodes: FxIndexSet<Node>,
}

pub struct VarSchedule {
    pub def_region: Region,
    pub def_pos: DefPos,
    pub last_use_pos: Option<UsePos>,
}

// FIXME(eddyb) this could be more space-efficient using appropriate niches.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum DefPos {
    RegionInput,
    NodeOutput { node_sched_idx: u32 },
}

// FIXME(eddyb) this could be more space-efficient using appropriate niches.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum UsePos {
    NodeInput { node_sched_idx: u32 },
    RegionOutput,

    // HACK(eddyb) `Loop` repeat conditions are effectively extra body outputs.
    LoopRepeatCond,
}

impl Schedule {
    pub fn compute(func_at_region: FuncAt<'_, Region>) -> Self {
        let mut sched = Schedule { regions: Default::default(), vars: Default::default() };

        let func = func_at_region.at(());

        let mut queue = VecDeque::new();
        queue.push_back((func_at_region.position, None));
        while let Some((region, parent_node)) = queue.pop_front() {
            let func_at_region = func.at(region);
            let region_def = func_at_region.def();

            sched.add_defs(&region_def.inputs, region, DefPos::RegionInput);

            let mut region_sched = RegionSchedule { nodes: Default::default() };
            for func_at_node in func_at_region.at_children() {
                let node = func_at_node.position;
                let node_sched_idx = u32::try_from(region_sched.nodes.len()).unwrap();
                // FIXME(eddyb) update uses of `insert` (e.g. in `spirti`) to `assert!`
                // uniqueness like this (maybe using a new method that always does it?).
                assert!(region_sched.nodes.insert(node));

                let node_def = func_at_node.def();

                sched.add_uses(&node_def.inputs, region, UsePos::NodeInput { node_sched_idx });
                sched.add_defs(&node_def.outputs, region, DefPos::NodeOutput { node_sched_idx });
                queue.extend(node_def.child_regions.iter().map(|&r| (r, Some(node))));
            }

            sched.add_uses(&region_def.outputs, region, UsePos::RegionOutput);

            if let Some(parent_node) = parent_node {
                // HACK(eddyb) `Loop` repeat conditions are effectively extra body outputs.
                if let NodeKind::Loop { repeat_condition } = func.nodes[parent_node].kind {
                    sched.add_uses(&[repeat_condition], region, UsePos::LoopRepeatCond);
                }
            }

            // FIXME(eddyb) update uses of `insert` (e.g. in `spirti`) to `assert!`
            // uniqueness like this (maybe using a new method that always does it?).
            assert!(sched.regions.insert(region, region_sched).is_none());
        }

        sched
    }

    fn add_defs(&mut self, defined_vars: &[Var], def_region: Region, def_pos: DefPos) {
        for &var in defined_vars {
            // FIXME(eddyb) update uses of `insert` (e.g. in `spirti`) to `assert!`
            // uniqueness like this (maybe using a new method that always does it?).
            assert!(
                self.vars
                    .insert(var, VarSchedule { def_region, def_pos, last_use_pos: None })
                    .is_none()
            );
        }
    }

    fn add_uses(&mut self, used_values: &[Value], use_region: Region, use_pos: UsePos) {
        for &v in used_values {
            match v {
                Value::Const(_) => {}
                Value::Var(var) => {
                    let var_sched = &mut self.vars[var];
                    assert!(
                        var_sched.def_region == use_region,
                        "cross-region use found, `cf::hermetic::seal` must be applied first!"
                    );

                    // FIXME(eddyb) consider asserting monotonicity here.
                    var_sched.last_use_pos = Some(use_pos);
                }
            }
        }
    }
}

pub trait RegBank: Copy + Eq + 'static {
    const ALL: &'static [Self];

    fn index(self) -> usize;
}

#[derive(Copy, Clone)]
pub struct RegName<B: RegBank> {
    pub bank: B,
    pub idx: u32,
}

/// On-the-fly simplistic (and conservative) register allocator, relying on
/// structured control-flow being visited in the same order as a [`Schedule`].
pub struct OnlineRegAlloc<'a, B: RegBank> {
    // TODO(eddyb) enforce that this is actually being traversed!
    sched: &'a Schedule,

    banks: Box<[RegBankState]>,
    vars: EntityOrientedDenseMap<Var, VarState<B>>,
}

#[derive(Default)]
struct RegBankState {
    regs: Vec<RegState>,
    last_freed: Option<u32>,
    last_temp: Option<u32>,
}

// FIXME(eddyb) consider using generation counters to additionally validate
// that reads aren't using some stale reference to the register.
#[derive(Copy, Clone, PartialEq, Eq)]
enum RegState {
    Free {
        // HACK(eddyb) instead of `Option`, this encodes `None` as its own index.
        next_free: u32,
    },

    // FIXME(eddyb) the need for `prev_temp` invalidates some of the rest of this
    // `enum`'s attempts at staying small, but is necessary in order to be able
    // to promote arbitrary registers
    Temp {
        // HACK(eddyb) instead of `Option`, this encodes `None` as its own index.
        prev_temp: u32,
        // HACK(eddyb) instead of `Option`, this encodes `None` as its own index.
        next_temp: u32,
    },

    OwnedBy(Var),

    // FIXME(eddyb) use this state to validate that the register is never written
    // to, until it returns to being `OwnedBy` some other `Var`.
    AliasedBy(Var),

    // HACK(eddyb) quasi-constant, never written during execution, attempts to
    // alias do not change this into `AliasedBy(_)`, but keep it `Pinned`.
    Pinned,
}

impl RegBankState {
    fn update_temp<R>(
        &mut self,
        idx: u32,
        f: impl FnOnce(&mut Option<u32>, &mut Option<u32>) -> R,
    ) -> R {
        let RegState::Temp { prev_temp, next_temp } = &mut self.regs[idx as usize] else {
            unreachable!();
        };

        let [mut maybe_prev_temp, mut maybe_next_temp] =
            [*prev_temp, *next_temp].map(|link| (link != idx).then_some(link));

        let r = f(&mut maybe_prev_temp, &mut maybe_next_temp);

        *prev_temp = maybe_prev_temp.unwrap_or(idx);
        *next_temp = maybe_next_temp.unwrap_or(idx);

        r
    }

    fn repurpose_temp(&mut self, idx: u32, new_state: RegState) {
        let reg_state = &mut self.regs[idx as usize];
        match *reg_state {
            RegState::Temp { prev_temp, next_temp } => {
                *reg_state = new_state;

                // FIXME(eddyb) consider implementing this logic via `update_temp`.
                let [prev_temp, next_temp] =
                    [prev_temp, next_temp].map(|link| (link != idx).then_some(link));

                if let Some(next_temp) = next_temp {
                    self.update_temp(next_temp, |prev_of_next, _| *prev_of_next = prev_temp);
                }
                if let Some(prev_temp) = prev_temp {
                    self.update_temp(prev_temp, |_, next_of_prev| *next_of_prev = next_temp);
                } else {
                    assert_eq!(self.last_temp, Some(idx));
                    self.last_temp = next_temp;
                }
            }

            RegState::Free { .. } => unreachable!("register already freed"),
            RegState::OwnedBy(_) | RegState::AliasedBy(_) => unreachable!("register already owned"),
            RegState::Pinned => unreachable!("register already pinned"),
        }
    }
}

struct VarState<B: RegBank> {
    reg_bank: B,
    reg_idx: u32,

    // HACK(eddyb) instead of `Option`, this encodes `None` as the same `Var`
    // (this is effectively a "next node" link in a linked list, with the head
    // being `banks[reg_bank].regs[reg_idx]` containing `RegState::AliasedBy`).
    reg_aliased_from: Var,
}

impl<'a, B: RegBank> OnlineRegAlloc<'a, B> {
    pub fn new(sched: &'a Schedule) -> Self {
        OnlineRegAlloc {
            sched,

            banks: B::ALL.iter().map(|_| RegBankState::default()).collect(),
            vars: Default::default(),
        }
    }

    pub fn used_banks_with_reg_counts(&self) -> impl Iterator<Item = (B, NonZeroU32)> {
        B::ALL
            .iter()
            .zip_eq(&self.banks)
            .map(|(&bank, bank_state)| (bank, u32::try_from(bank_state.regs.len()).unwrap()))
            .filter_map(|(bank, count)| Some((bank, NonZeroU32::new(count)?)))
    }

    pub fn alloc_temp(&mut self, bank: B) -> RegName<B> {
        let bank_state = &mut self.banks[bank.index()];

        let last_freed = bank_state.last_freed.take();
        let last_temp = bank_state.last_temp.take();

        let idx = last_freed.unwrap_or_else(|| u32::try_from(bank_state.regs.len()).unwrap());
        let reg_state = RegState::Temp { prev_temp: idx, next_temp: last_temp.unwrap_or(idx) };

        if let Some(idx) = last_freed {
            let RegState::Free { next_free } =
                mem::replace(&mut bank_state.regs[idx as usize], reg_state)
            else {
                unreachable!();
            };
            if next_free != idx {
                bank_state.last_freed = Some(next_free);
            }
        } else {
            bank_state.regs.push(reg_state);
        }

        if let Some(last_temp) = last_temp {
            bank_state.update_temp(last_temp, |prev_of_last, _| *prev_of_last = Some(idx));
        }

        bank_state.last_temp = Some(idx);

        RegName { bank, idx }
    }

    // FIXME(eddyb) is there a better solution for "quasi-global registers"?
    pub fn pin_temp(&mut self, temp_reg: RegName<B>) {
        self.banks[temp_reg.bank.index()].repurpose_temp(temp_reg.idx, RegState::Pinned);
    }

    pub fn use_var(&self, var: Var) -> RegName<B> {
        let VarState { reg_bank, reg_idx, .. } = self.vars[var];

        // FIXME(eddyb) this could/should check some kind of "generation counter".

        RegName { bank: reg_bank, idx: reg_idx }
    }

    pub fn def_var(&mut self, var: Var, bank: B) -> RegName<B> {
        // FIXME(eddyb) could be more efficient by avoiding the temp list.
        let reg = self.alloc_temp(bank);
        self.def_var_taking_ownership_of_temp(var, reg);
        reg
    }

    pub fn def_var_taking_ownership_of_temp(&mut self, var: Var, temp_reg: RegName<B>) {
        let bank_state = &mut self.banks[temp_reg.bank.index()];

        let var_state = self.vars.entry(var);
        assert!(var_state.is_none());

        bank_state.repurpose_temp(temp_reg.idx, RegState::OwnedBy(var));

        *var_state = Some(VarState {
            reg_bank: temp_reg.bank,
            reg_idx: temp_reg.idx,
            reg_aliased_from: var,
        });
    }

    pub fn def_var_aliasing(&mut self, var: Var, aliased_reg: RegName<B>) {
        let bank_state = &mut self.banks[aliased_reg.bank.index()];

        let var_state = self.vars.entry(var);
        assert!(var_state.is_none());

        let reg_state = &mut bank_state.regs[aliased_reg.idx as usize];
        let reg_aliased_from = match *reg_state {
            RegState::Free { .. } => unreachable!("register already freed"),
            RegState::Temp { .. } => unreachable!("register was never owned"),
            RegState::OwnedBy(prev_var) | RegState::AliasedBy(prev_var) => {
                *reg_state = RegState::AliasedBy(var);
                prev_var
            }
            RegState::Pinned => var,
        };

        *var_state = Some(VarState {
            reg_bank: aliased_reg.bank,
            reg_idx: aliased_reg.idx,
            reg_aliased_from,
        });
    }

    pub fn release_var(&mut self, var: Var) {
        self.maybe_release_var(var).expect("var was never defined");
    }

    pub fn release_var_if_defined(&mut self, var: Var) {
        let _ = self.maybe_release_var(var);
    }

    fn maybe_release_var(&mut self, var: Var) -> Option<()> {
        // FIXME(eddyb) should this use a tombstone instead?
        let VarState { reg_bank, reg_idx, reg_aliased_from } = self.vars.remove(var)?;

        let bank_state = &mut self.banks[reg_bank.index()];
        let reg_state = &mut bank_state.regs[reg_idx as usize];
        if reg_aliased_from == var {
            if *reg_state != RegState::Pinned {
                assert!(*reg_state == RegState::OwnedBy(var));
                *reg_state =
                    RegState::Free { next_free: bank_state.last_freed.take().unwrap_or(reg_idx) };
                bank_state.last_freed = Some(reg_idx);
            }
        } else {
            assert!(*reg_state == RegState::AliasedBy(var));
            let prev_var = reg_aliased_from;
            *reg_state = if self.vars[prev_var].reg_aliased_from == prev_var {
                RegState::OwnedBy(prev_var)
            } else {
                RegState::AliasedBy(prev_var)
            };
        }

        Some(())
    }

    pub fn release_temps(&mut self) {
        for bank_state in &mut self.banks {
            while let Some(reg_idx) = bank_state.last_temp.take() {
                let reg_state = &mut bank_state.regs[reg_idx as usize];
                match *reg_state {
                    RegState::Temp { prev_temp: _, next_temp } => {
                        *reg_state = RegState::Free {
                            next_free: bank_state.last_freed.take().unwrap_or(reg_idx),
                        };
                        bank_state.last_freed = Some(reg_idx);

                        if next_temp != reg_idx {
                            bank_state.last_temp = Some(next_temp);
                        }
                    }

                    _ => unreachable!(),
                }
            }
        }
    }
}
