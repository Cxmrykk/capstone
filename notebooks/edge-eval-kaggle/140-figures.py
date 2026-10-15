# 13. Figures for the thesis (PNG at 300 dpi + vector PDF)
import matplotlib.pyplot as plt

FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 9})


def save_fig(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG_DIR / (name + '.' + ext), bbox_inches='tight')
    plt.show()
    plt.close(fig)


def _short(jid):
    return jid.replace('__fine_tuned', '').replace('__gguf_', LF).replace('_cpu', '').replace('_prompts', ' prompts')


# Figure 1: execution accuracy, GPU 4-bit vs CPU GGUF on identical rows (Wilson CIs).
if not OVERALL.empty and OVERALL['execution_accuracy'].notna().any():
    sub = OVERALL.dropna(subset=['execution_accuracy'])
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(sub)), 4))
    for k, (prefix, label) in enumerate((('gpu_', 'GPU 4-bit HF (same rows)'), ('', 'CPU GGUF'))):
        col = prefix + 'execution_accuracy'
        if col not in sub.columns:
            continue
        vals = sub[col].to_numpy(float)
        lo = sub.get(col + '_ci_low', sub[col]).to_numpy(float)
        hi = sub.get(col + '_ci_high', sub[col]).to_numpy(float)
        yerr = np.clip(np.nan_to_num(np.vstack([vals - lo, hi - vals])), 0, None)
        ax.bar(x + (k - 0.5) * 0.38, np.nan_to_num(vals), 0.38, yerr=yerr, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(j) for j in sub['job_id']], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Execution accuracy')
    ax.set_title('Execution accuracy on identical rows: GPU vs CPU (' + str(int(CI_LEVEL * 100)) + '% CI)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    save_fig(fig, 'fig1_edge_vs_gpu_execution_accuracy')

# Figure 2: per-query latency, median prefill + decode, with p95 marker.
if not OVERALL.empty and 'prefill_ms_p50' in OVERALL.columns:
    sub = OVERALL.dropna(subset=['wall_ms_p50'])
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(sub) + 1.5))
    pre = sub['prefill_ms_p50'].to_numpy(float) / 1000
    dec = sub['decode_ms_p50'].to_numpy(float) / 1000
    ax.barh(y, pre, label='prefill (median)')
    ax.barh(y, dec, left=pre, label='decode (median)')
    ax.scatter(sub['wall_ms_p95'].to_numpy(float) / 1000, y, marker='|', s=200, color='black', label='p95 total')
    ax.set_yticks(y)
    ax.set_yticklabels([_short(j).replace(LF, ' ') for j in sub['job_id']], fontsize=7)
    ax.set_xlabel('Seconds per query (' + CPU_MODEL[:40] + ')')
    ax.set_title('Per-query latency on the CPU')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=7)
    save_fig(fig, 'fig2_latency_breakdown')

# Figure 3: llama-bench throughput vs threads.
if not BENCH.empty and 'tokens_per_s' in BENCH.columns:
    sweep = BENCH[BENCH['quant'].isin(THREAD_SWEEP_QUANTS) & BENCH['tokens_per_s'].notna()]
    if not sweep.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, kind in zip(axes, ('pp', 'tg')):
            for (model, quant), g in sweep[sweep['test'].str.startswith(kind)].groupby(['model', 'quant']):
                g = g.sort_values('threads')
                ax.errorbar(g['threads'], g['tokens_per_s'], yerr=g['tokens_per_s_sd'], marker='o', capsize=3,
                            label=model + ' ' + quant)
            ax.set_xlabel('Threads')
            ax.set_ylabel('Tokens/s')
            ax.set_title(('Prompt processing (pp' + str(BENCH_PROMPT_TOKENS) + ')') if kind == 'pp'
                         else ('Generation (tg' + str(BENCH_GEN_TOKENS) + ')'))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        save_fig(fig, 'fig3_throughput_vs_threads')

# Figure 4: quantisation trade-off: file size vs generation speed (full thread count).
if not BENCH.empty and 'tokens_per_s' in BENCH.columns:
    tg = BENCH[BENCH['test'].str.startswith('tg') & BENCH['threads'].eq(LOGICAL_CPUS)].dropna(subset=['tokens_per_s'])
    if not tg.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, g in tg.groupby('model'):
            ax.plot(g['gguf_size_mb'], g['tokens_per_s'], marker='o', label=model)
            for _, r in g.iterrows():
                ax.annotate(r['quant'], (r['gguf_size_mb'], r['tokens_per_s']), fontsize=7,
                            xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel('GGUF size (MB)')
        ax.set_ylabel('Generation tokens/s (' + str(LOGICAL_CPUS) + ' threads)')
        ax.set_title('Quantisation: size vs generation speed')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        save_fig(fig, 'fig4_quant_size_vs_speed')

# Figure 5: prefill time vs prompt tokens (why schema filtering matters on the CPU).
if not EDGE_LONG.empty and 'edge_prompt_ms' in EDGE_LONG.columns:
    pts = EDGE_LONG[EDGE_LONG['quant'] == ACCURACY_QUANT]
    if not pts.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (model, mode), g in pts.groupby(['model', 'prompt_schema_mode']):
            ax.scatter(pd.to_numeric(g['edge_tokens_evaluated'], errors='coerce'),
                       pd.to_numeric(g['edge_prompt_ms'], errors='coerce') / 1000, s=8, alpha=0.6,
                       label=model + ' / ' + mode)
        ax.set_xlabel('Prompt tokens')
        ax.set_ylabel('Prefill seconds')
        ax.set_title('Prefill cost vs prompt length (' + ACCURACY_QUANT + ')')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        save_fig(fig, 'fig5_prefill_vs_prompt_tokens')

# Figure 6: exact-match filter coverage of gold-query elements by data source.
if not COVERAGE.empty:
    sub = COVERAGE.set_index('group').sort_values('n_parsed', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(sub) + 1.5))
    sub[['share_fully_covered', 'share_missing_label', 'share_missing_rel_type', 'share_missing_property']].plot(
        kind='barh', ax=ax, width=0.8)
    ax.set_xlim(0, 1)
    ax.set_yticklabels([g + ' (n=' + str(int(n)) + ')' for g, n in zip(sub.index, sub['n_parsed'])], fontsize=7)
    ax.set_xlabel('Share of rows')
    ax.set_title('Exact-match filter: gold-query elements kept in the prompt schema')
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    save_fig(fig, 'fig6_filter_coverage')

# Figure 7: schema hallucination per job.
if not HALLUCINATION.empty:
    sub = HALLUCINATION.sort_values(['source', 'job_id'])
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(sub) + 1.5))
    sub.set_index('job_id')[['share_unknown_label', 'share_unknown_rel_type', 'share_unknown_property']].plot(
        kind='barh', ax=ax, width=0.8)
    ax.set_xlabel('Share of rows with a non-empty prediction')
    ax.set_title('Predicted schema elements absent from the full schema')
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(fontsize=7)
    save_fig(fig, 'fig7_schema_hallucination')

# Figure 8: accuracy vs memory footprint on the CPU.
if not OVERALL.empty and OVERALL['peak_rss_mb'].notna().any() and OVERALL['execution_accuracy'].notna().any():
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, r in OVERALL.dropna(subset=['peak_rss_mb', 'execution_accuracy']).iterrows():
        ax.scatter(r['peak_rss_mb'], r['execution_accuracy'])
        ax.annotate(_short(r['job_id']).replace(LF, ' '), (r['peak_rss_mb'], r['execution_accuracy']), fontsize=6,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Peak resident memory of llama-server (MB)')
    ax.set_ylabel('Execution accuracy (CPU)')
    ax.set_title('Accuracy vs memory footprint')
    ax.grid(alpha=0.3)
    save_fig(fig, 'fig8_accuracy_vs_memory')

print('Figures saved to ' + str(FIG_DIR))
