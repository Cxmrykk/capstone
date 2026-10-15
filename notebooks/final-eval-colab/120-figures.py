# 11. Figures for the thesis (PNG at 300 dpi + vector PDF)
import matplotlib.pyplot as plt

FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 9})


def save_fig(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG_DIR / (name + '.' + ext), bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Figures 1-3: headline metrics, zero-shot vs fine-tuned, with CIs.
hf = OVERALL[OVERALL['backend'] == 'hf'].copy()
hf['label'] = hf['model'] + LF + hf['schema_mode']
labels = list(dict.fromkeys(hf['label']))
x = np.arange(len(labels))
width = 0.38
for metric, title, name in (('execution_accuracy', 'Execution accuracy', 'fig1_execution_accuracy'),
                            ('google_bleu_corpus', 'Google-BLEU (corpus)', 'fig2_google_bleu'),
                            ('exact_match', 'Exact match', 'fig3_exact_match')):
    if metric not in hf.columns or hf[metric].isna().all():
        continue
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4))
    for k, cond in enumerate(('zero_shot', 'fine_tuned')):
        sub = hf[hf['condition'] == cond].set_index('label').reindex(labels)
        if sub[metric].isna().all():
            continue
        vals = sub[metric].to_numpy(float)
        lo = sub[metric + '_ci_low'].to_numpy(float) if metric + '_ci_low' in sub.columns else vals
        hi = sub[metric + '_ci_high'].to_numpy(float) if metric + '_ci_high' in sub.columns else vals
        yerr = np.nan_to_num(np.vstack([vals - lo, hi - vals]))
        ax.bar(x + (k - 0.5) * width, np.nan_to_num(vals), width, yerr=yerr, capsize=3,
               label=cond.replace('_', '-'))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel(title)
    ax.set_title(title + ' (' + SPLIT + ' split, n=' + LIMIT_LABEL + ', ' + str(int(CI_LEVEL * 100)) + '% CI)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    save_fig(fig, name)

# Figure 4: execution accuracy by query complexity (fine-tuned models).
ft = BY_COMPLEXITY[(BY_COMPLEXITY['condition'] == 'fine_tuned') & (BY_COMPLEXITY['backend'] == 'hf')]
if not ft.empty and 'execution_accuracy' in ft.columns and ft['execution_accuracy'].notna().any():
    pv = ft.pivot_table(index='complexity', columns='run_name', values='execution_accuracy',
                        aggfunc='first').reindex(COMPLEXITY_ORDER)
    counts = ft.groupby('complexity')['n_gold_ok'].max().reindex(COMPLEXITY_ORDER) if 'n_gold_ok' in ft.columns else None
    fig, ax = plt.subplots(figsize=(8, 4))
    pv.plot(kind='bar', ax=ax, rot=0, width=0.8)
    ticks = []
    for c in COMPLEXITY_ORDER:
        v = counts.get(c) if counts is not None else None
        ticks.append(c + LF + 'n=' + (str(int(v)) if v is not None and pd.notna(v) else '0'))
    ax.set_xticklabels(ticks)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Execution accuracy')
    ax.set_xlabel('Query complexity (gold query)')
    ax.set_title('Fine-tuned execution accuracy by query complexity')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    save_fig(fig, 'fig4_execution_accuracy_by_complexity')

# Figure 5: training and validation loss curves.
if not CURVES.empty and 'loss' in CURVES.columns:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for rn, g in CURVES.groupby('run_name'):
        tr = g[g['kind'] == 'train']
        if not tr.empty:
            axes[0].plot(tr['step'], tr['loss'], label=rn, linewidth=1)
        if 'eval_loss' in g.columns:
            ev = g[g['kind'] == 'eval']
            if not ev.empty:
                axes[1].plot(ev['step'], ev['eval_loss'], marker='o', markersize=3, label=rn, linewidth=1)
    axes[0].set_title('Training loss')
    axes[1].set_title('Validation loss')
    for ax in axes:
        ax.set_xlabel('Optimizer step')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    save_fig(fig, 'fig5_training_curves')

# Figure 6: prompt length under each schema mode.
tok = ROWS[ROWS['backend'] == 'hf'].drop_duplicates(['model', 'schema_mode', 'instance_id'])
groups = [(m + LF + s, g['prompt_tokens'].dropna().to_numpy()) for (m, s), g in tok.groupby(['model', 'schema_mode'])]
groups = [gr for gr in groups if len(gr[1])]
if groups:
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(groups)), 4))
    ax.boxplot([g for _, g in groups], showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([lbl for lbl, _ in groups])
    ax.set_ylabel('Prompt tokens')
    ax.set_title('Prompt length by schema mode (outliers hidden)')
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, 'fig6_prompt_tokens')

# Figure 7: error categories per job.
cats = [c for c in ('correct', 'wrong_result', 'invalid_query', 'runtime_error', 'execution_error',
                    'empty_output', 'gold_failed', 'not_executed') if c in ERRORS.columns]
if cats:
    share = ERRORS.set_index('job_id')[cats]
    share = share.div(share.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(share) + 1.5))
    share.plot(kind='barh', stacked=True, ax=ax, width=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Share of rows')
    ax.set_title('Outcome categories per job')
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    save_fig(fig, 'fig7_error_categories')

print('Figures saved to ' + str(FIG_DIR))
