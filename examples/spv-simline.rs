use std::collections::HashMap;
use std::ops::Range;

struct FuncInfo {
    word_range: Range<usize>,
    inst_count: usize,

    calls: Vec<FuncRef>,
    call_inst_word_count: usize,

    name: Option<String>,

    post_simulated_inlining_stats: Option<Result<FuncStats, RecursionDetected>>,
    simulated_inlining_incoming_call_count: usize,
}

#[derive(Copy, Clone, Debug)]
struct RecursionDetected;

#[derive(Copy, Clone)]
struct FuncStats {
    word_count: usize,
    inst_count: usize,
}

struct FuncRef {
    id: u32,
    resolved_func_idx: Option<usize>,
}

#[derive(Default)]
struct State {
    op_name_word_range: Option<Range<usize>>,
    entry_points: Vec<(FuncRef, String)>,

    func_id_to_idx: HashMap<u32, usize>,
    funcs: Vec<FuncInfo>,
    in_func: bool,
}

fn nul_terminated_utf8_str_from_words(it: impl Iterator<Item = u32>) -> String {
    String::from_utf8(it.flat_map(u32::to_le_bytes).take_while(|&x| x != 0).collect::<Vec<u8>>())
        .unwrap()
}

fn insts_from_words(
    word_range: Range<usize>,
    get_word: impl Fn(usize) -> u32,
) -> impl Iterator<Item = (u16, Range<usize>)> {
    let mut next_inst_start = word_range.start;
    std::iter::from_fn(move || {
        if next_inst_start >= word_range.end {
            return None;
        }
        let inst_start = next_inst_start;
        let op_and_len = get_word(inst_start);
        let (op, len) = (op_and_len as u16, (op_and_len >> 16) as usize);
        next_inst_start += len;

        Some((op, inst_start..next_inst_start))
    })
}

impl State {
    fn inst(&mut self, op: u16, word_range: Range<usize>, get_word: impl Fn(usize) -> u32) {
        match op {
            5 => {
                assert!(!self.in_func);
                self.op_name_word_range.get_or_insert(word_range.clone()).end = word_range.end;
            }
            15 => {
                assert!(!self.in_func);
                self.entry_points.push((
                    FuncRef { id: get_word(word_range.start + 2), resolved_func_idx: None },
                    nul_terminated_utf8_str_from_words(word_range.clone().skip(3).map(get_word)),
                ));
            }
            54 => {
                assert!(!self.in_func);
                let func_id = get_word(word_range.start + 2);
                let func_idx = self.funcs.len();
                self.funcs.push(FuncInfo {
                    word_range,
                    inst_count: 0,
                    calls: vec![],
                    call_inst_word_count: 0,

                    name: None,
                    post_simulated_inlining_stats: None,
                    simulated_inlining_incoming_call_count: 0,
                });
                self.func_id_to_idx.insert(func_id, func_idx);
                self.in_func = true;
                return;
            }
            56 => {
                assert!(self.in_func);
                self.funcs.last_mut().unwrap().word_range.end = word_range.end;
                self.in_func = false;
            }
            57 => {
                assert!(self.in_func);
                let func = self.funcs.last_mut().unwrap();
                func.calls
                    .push(FuncRef { id: get_word(word_range.start + 3), resolved_func_idx: None });
                func.call_inst_word_count += word_range.len();
            }
            _ => {}
        }
        if self.in_func {
            let func = self.funcs.last_mut().unwrap();
            func.word_range.end = word_range.end;
            func.inst_count += 1;
        }
    }

    fn simulate_inlining(&mut self, func_idx: usize) -> FuncStats {
        let func = &mut self.funcs[func_idx];
        if let Some(stats) = func.post_simulated_inlining_stats.transpose().unwrap() {
            return stats;
        }
        func.post_simulated_inlining_stats = Some(Err(RecursionDetected));

        let mut stats = FuncStats {
            word_count: func.word_range.len() - func.call_inst_word_count,
            inst_count: func.inst_count - func.calls.len(),
        };
        for call_idx in 0..func.calls.len() {
            let callee_idx = self.funcs[func_idx].calls[call_idx].resolved_func_idx.unwrap();
            self.funcs[callee_idx].simulated_inlining_incoming_call_count += 1;
            let callee_stats = self.simulate_inlining(callee_idx);
            stats.word_count += callee_stats.word_count;
            stats.inst_count += callee_stats.inst_count;
        }
        self.funcs[func_idx].post_simulated_inlining_stats = Some(Ok(stats));
        stats
    }

    fn finish(mut self, get_word: impl Fn(usize) -> u32) {
        assert!(!self.in_func);

        for func_ref in (self.entry_points.iter_mut().map(|(r, _)| r))
            .chain(self.funcs.iter_mut().flat_map(|func| &mut func.calls))
        {
            func_ref.resolved_func_idx = Some(self.func_id_to_idx[&func_ref.id]);
        }

        if let Some(op_name_word_range) = self.op_name_word_range.clone() {
            for (op, word_range) in insts_from_words(op_name_word_range, &get_word) {
                if op == 5
                    && let Some(&func_idx) =
                        self.func_id_to_idx.get(&get_word(word_range.start + 1))
                {
                    self.funcs[func_idx].name.get_or_insert_with(|| {
                        nul_terminated_utf8_str_from_words(
                            word_range.clone().skip(2).map(&get_word),
                        )
                    });
                }
            }
        }
        for (func_ref, name) in &self.entry_points {
            self.funcs[func_ref.resolved_func_idx.unwrap()]
                .name
                .get_or_insert_with(|| name.clone());
        }

        for func_idx in 0..self.funcs.len() {
            self.simulate_inlining(func_idx);
        }

        if self.entry_points.len() == 1 && true {
            let mib_of = |func_idx: usize| {
                ((self.funcs[func_idx].post_simulated_inlining_stats.unwrap().unwrap().word_count
                    * 4) as f64)
                    / (1024.0 * 1024.0)
            };

            println!(
                "{:.3} MiB vs {:.3} MiB",
                mib_of(self.entry_points[0].0.resolved_func_idx.unwrap()),
                (0..self.funcs.len()).map(mib_of).sum::<f64>()
            );

            return;
        }

        let min_word_count_contrib = self
            .entry_points
            .iter()
            .map(|(func_ref, _)| {
                self.funcs[func_ref.resolved_func_idx.unwrap()]
                    .post_simulated_inlining_stats
                    .unwrap()
                    .unwrap()
                    .word_count
            })
            .max()
            .unwrap()
            / 1000;
        let include_func = |func_idx: usize| {
            self.funcs[func_idx].post_simulated_inlining_stats.unwrap().unwrap().word_count
                >= min_word_count_contrib
        };

        println!("digraph {{");
        println!("  node [shape=box, fontname=monospace];");
        for (caller_idx, caller) in self.funcs.iter().enumerate() {
            if !include_func(caller_idx) {
                continue;
            }

            let name = caller.name.as_deref().unwrap_or("<unnamed function>");
            let post_simulated_inlining_stats =
                caller.post_simulated_inlining_stats.unwrap().unwrap();
            let label = format!(
                "{name} [Total: {} calls, {} insts / {:.3} MiB each | Self: {:.3} MiB]",
                caller.simulated_inlining_incoming_call_count,
                post_simulated_inlining_stats.inst_count,
                ((post_simulated_inlining_stats.word_count * 4) as f64) / (1024.0 * 1024.0),
                ((caller.word_range.len() * 4) as f64) / (1024.0 * 1024.0)
            );
            println!("  f{caller_idx} [label={label:?}];");
            for callee in &caller.calls {
                let callee_idx = callee.resolved_func_idx.unwrap();
                if !include_func(callee_idx) {
                    continue;
                }
                println!("    f{caller_idx} -> f{callee_idx};");
            }
        }
        println!("}}");
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let [_, spv_file] = &args[..] else {
        eprintln!("Usage: {} FILE", args[0]);
        std::process::exit(1);
    };

    let spv_bytes = std::fs::read(spv_file).unwrap();
    let (spv_words, &[]) = spv_bytes.as_chunks() else {
        eprintln!("`{spv_file}` has {} bytes (expected a multiple of 4)", spv_bytes.len());
        std::process::exit(1);
    };
    let spv_word = |i| u32::from_le_bytes(spv_words[i]);
    assert_eq!(spv_word(0), 0x07230203);
    assert_eq!(spv_word(4), 0);

    let mut state = State::default();
    for (op, word_range) in insts_from_words(5..spv_words.len(), spv_word) {
        state.inst(op, word_range, spv_word)
    }
    state.finish(spv_word);
}
