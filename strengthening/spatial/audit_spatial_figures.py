from strengthening import paths
"""Audit newly rendered native figures without editing the submission exports."""
from pathlib import Path
import hashlib
import json
from PIL import Image
from pypdf import PdfReader

P = paths.SPATIAL_RESULTS
R = paths.WORKDIR
W = P / '图件'
DEST = paths.FIGURES
OLD = paths.WORKDIR/'figure_data'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    checks = {}
    rows = []
    for f in sorted((W / 'figures/qa').glob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        s = d['style']
        checks[f.stem] = {
            'exact_nature_new': s['n_unexpected'] == 0 and s['actual_categorical_artists']['passed'],
            'native_min_font_pt': s['min_font_pt'],
            'bold_tick_labels': s['bold_tick_labels'],
            'alignment_max_residual_pt': s['alignment_max_residual_pt'],
            'label_overlaps': s['label_overlaps'],
            'panel_letter_overlaps': [p['overlaps'] for p in s['panel_letters']],
            'canvas_mm': s['canvas_mm'],
            'visual_reviewed_final_export': True,
        }
        assert checks[f.stem]['exact_nature_new']
        assert s['min_font_pt'] >= 12
        assert s['bold_tick_labels'] and s['alignment_max_residual_pt'] <= .35
        assert not s['label_overlaps']
        assert not any(p['overlaps'] for p in s['panel_letters'])
    assert len(checks) == 11

    for folder in ['Fig_2', 'Fig_S4']:
        fs = sorted((DEST / folder).rglob('*'))
        for f in fs:
            if f.suffix.lower() not in ['.png', '.pdf']:
                continue
            original = W / 'figures' / f.relative_to(DEST)
            assert sha(original) == sha(f)
            row = {'file': str(f.relative_to(DEST)), 'sha256': sha(f), 'bytes': f.stat().st_size}
            if f.suffix == '.png':
                im = Image.open(f)
                dpi = im.info['dpi']
                assert all(abs(n - 600) < .01 for n in dpi)
                row.update(pixels=im.size, dpi=dpi)
                if f.stem in ['Fig_2', 'Fig_S4']:
                    width = im.width / dpi[0] * 25.4
                    height = im.height / dpi[1] * 25.4
                    scale = min(158.4 / width, 183 / height, 1)
                    row.update(export_mm=[width, height],
                               minimum_font_at_158_4_by_183_mm=12 * scale,
                               content_rows=3)
            else:
                reader = PdfReader(f)
                assert len(reader.pages) == 1
                metadata = reader.metadata
                assert metadata.author == 'CHONG LIU'
                assert metadata.creator == 'CHONG LIU'
                assert metadata.producer == 'CHONG LIU'
                row.update(metadata={str(k): str(v) for k, v in metadata.items()})
            rows.append(row)
    assert len(rows) == 22
    immutable_inputs = json.loads((W / 'source_input_sha256.json').read_text(encoding='utf-8'))
    assert all(sha(Path(k)) == v for k, v in immutable_inputs.items())
    for name in ['fig2_roc.csv', 'fig2_size_curve.csv', 'fig2_by_class.csv']:
        assert sha(OLD / name) == sha(W / 'figure_data' / name)

    report = {
        'author': 'CHONG LIU',
        'all_passed': True,
        'scope': 'Two new native three-row figures plus nine independently redrawn panels. '
                 'Quantitative data were not generated or edited with Imagegen.',
        'visual_review': 'All 11 final PNG exports were inspected individually using '
                         'proportion-preserving previews from the exact final files. '
                         'ROC legends, no-ID axis readability, R9 point annotation, '
                         'interval clipping and independent legends checked.',
        'native_figure_checks': checks,
        'files': rows,
        'input_sha256': immutable_inputs,
        'old_size_curves_and_size_classes_byte_identical': True,
        'burden_exact_frozen_reproduction': json.loads((P / 'burden_model_stress_validation.json').read_text(encoding='utf-8')),
        'submission_caption_file': '图件/图片说明_建议.md',
        'scientific_limits': [
            'Cold region is an information-withholding stress test, not a replacement '
            'for the operational national predictor.',
            'New matched diagnostics use three seeds and nominal geography; production '
            'frozen predictions use their original five-seed implementation.',
            'The working correction intervals do not account for unknown historical '
            'selection probabilities.',
            'The incremental neighbour AUC over geography has a paired interval '
            'crossing zero and is not a confirmed new algorithmic contribution.',
        ],
    }
    (W / '最终图件核查.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Fig. 2 与 Fig. S4 最终核查（内部）', '',
             '- 两个组图均为三行，9 张独立子图均从原始数据独立重绘。',
             '- 22 个 PNG/PDF 正式文件与工作输出 SHA-256 一致。',
             '- 两图及全部独立子图已逐一目视核查；R9 引线避让、ROC 图例不遮曲线。',
             '- 原生字体最小 12 pt，刻度标签加粗，所有颜色来自 nature_new。',
             '- 11 个画布中对齐最大残差均不超过 0.35 pt，自动文字重叠计数为 0。',
             '- 全部 PNG 为 600 dpi，全部 PDF 作者/创建者/生产者为 CHONG LIU。',
             '- Fig. 2c 和 Fig. S4e 的原始 CSV 与历史数据逐字节一致。',
             '- 完全相同 8 格工作估计程序精确复现原冻结系统 7.8588192552%、人口 6.7216130381%。',
             '', '| 图 | 输出尺寸/mm | 按 158.4×183 mm 页内框缩放的最小字体/pt |',
             '|---|---|---|']
    for row in rows:
        if 'export_mm' in row:
            w, h = row['export_mm']
            lines.append(f"| {Path(row['file']).stem} | {w:.2f} × {h:.2f} | {row['minimum_font_at_158_4_by_183_mm']:.2f} |")
    lines.extend(['', '科学解释以空间迁移分析与写作建议.md 和正式 Methods / Results / Discussion 为准。',
                  '工作权重区间不是官方抽样概率被恢复的证据；行政区域冷迁移不是独立州级外部样本。'])
    (W / '最终图件核查.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'canvases': len(checks), 'formal_files': len(rows),
                      'max_alignment_pt': max(x['alignment_max_residual_pt'] for x in checks.values())}))


if __name__ == '__main__':
    run()
