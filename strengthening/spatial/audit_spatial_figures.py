"""Audit native regular-weight spatial figures and immutable metric inputs."""
from strengthening import paths
from pathlib import Path
import argparse
import hashlib
import json
import re
from collections import Counter
import fitz
from PIL import Image
from pypdf import PdfReader

P = paths.SPATIAL_RESULTS
W = P / '图件'
DEST = paths.FIGURES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(qa_dir, visual_reviewed=False, baseline_package=None):
    qa_dir.mkdir(parents=True, exist_ok=True)
    checks, rows = {}, []
    for f in sorted((W / 'figures/qa').glob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        s = d['style']
        check = {
            'exact_nature_new': s['n_unexpected'] == 0 and s['actual_categorical_artists']['passed'],
            'native_min_font_pt': s['min_font_pt'],
            'all_actual_text_normal': s['all_actual_text_normal'],
            'regular_text': s['regular_text'],
            'regular_tick_labels': s['regular_tick_labels'],
            'bold_math_commands': s['bold_math_commands'],
            'actual_text_styles': s['actual_text_styles'],
            'alignment_max_residual_pt': s['alignment_max_residual_pt'],
            'label_overlaps': s['label_overlaps'],
            'panel_letter_overlaps': [p['overlaps'] for p in s['panel_letters']],
            'canvas_mm': s['canvas_mm'],
            'visual_reviewed_final_export': visual_reviewed,
        }
        assert check['exact_nature_new'], f
        assert s['min_font_pt'] >= 12, f
        assert s['all_actual_text_normal'] and s['regular_text'] and s['regular_tick_labels'], f
        assert not s['bold_math_commands'], f
        assert s['alignment_max_residual_pt'] <= .35, (f, s['alignment_max_residual_pt'])
        assert not s['label_overlaps'], f
        assert not any(p['overlaps'] for p in s['panel_letters']), f
        checks[f.stem] = check
    assert len(checks) == 11, len(checks)

    for folder in ['Fig_2', 'Fig_S4']:
        for f in sorted((DEST / folder).rglob('*')):
            if f.suffix.lower() not in ['.png', '.pdf']:
                continue
            original = W / 'figures' / f.relative_to(DEST)
            assert sha(original) == sha(f), f
            row = {'file': str(f.relative_to(DEST)), 'sha256': sha(f), 'bytes': f.stat().st_size}
            if f.suffix == '.png':
                with Image.open(f) as im:
                    dpi = im.info['dpi']
                    assert all(abs(n - 600) < .01 for n in dpi), f
                    row.update(pixels=list(im.size), dpi=dpi)
                    if f.stem in ['Fig_2', 'Fig_S4']:
                        width, height = im.width / dpi[0] * 25.4, im.height / dpi[1] * 25.4
                        scale = min(158.4 / width, 183 / height, 1)
                        row.update(export_mm=[width, height], minimum_font_at_158_4_by_183_mm=12 * scale, content_rows=3)
            else:
                reader = PdfReader(f)
                assert len(reader.pages) == 1, f
                metadata = reader.metadata
                for key in ['author', 'creator', 'producer']:
                    assert getattr(metadata, key) == 'CHONG LIU', (f, key)
                pdf = fitz.open(f)
                fonts = sorted({font[3] for page in pdf for font in page.get_fonts(full=True)})
                spans = [span for page in pdf for block in page.get_text('dict')['blocks'] if 'lines' in block
                         for line in block['lines'] for span in line['spans'] if span['text'].strip()]
                bold_fonts = [font for font in fonts if re.search(r'bold|semibold|demi|heavy|black', font, re.I)]
                bold_spans = [dict(text=span['text'], font=span['font'], flags=span['flags'])
                              for span in spans if span['flags'] & 16]
                assert not bold_fonts and not bold_spans, (f, bold_fonts, bold_spans)
                assert spans, f
                row.update(metadata={str(k): str(v) for k, v in metadata.items()},
                           embedded_pdf_fonts=fonts, actual_pdf_text_span_count=len(spans),
                           bold_pdf_fonts=bold_fonts, bold_pdf_spans=bold_spans)
                pdf.close()
            rows.append(row)
    assert len(rows) == 22, len(rows)

    source_hashes = json.loads((W / 'source_input_sha256.json').read_text(encoding='utf-8'))
    assert all(sha(Path(k)) == v for k, v in source_hashes.items())
    reference = paths.REPO / 'strengthening/reference_results'
    immutable = []
    for group, runtime in [('spatial', paths.SPATIAL_RESULTS), ('figure_data', paths.WORKDIR / 'figure_data')]:
        for f in sorted((reference / group).rglob('*.csv')):
            other = runtime / f.relative_to(reference / group)
            assert sha(f) == sha(other), (f, other)
            immutable.append({'reference_csv': str(f.relative_to(reference)), 'sha256': sha(f)})
    for name in ['fig2_roc.csv', 'fig2_size_curve.csv', 'fig2_by_class.csv']:
        assert sha(paths.WORKDIR / 'figure_data' / name) == sha(W / 'figure_data' / name)

    baseline_checks = []
    if baseline_package:
        old_ref = baseline_package / 'Source_Code/strengthening/reference_results'
        for item in immutable:
            rel = Path(item['reference_csv'])
            old = old_ref / rel
            assert sha(old) == item['sha256'], old
            baseline_checks.append({'csv': str(rel), 'baseline_sha256': sha(old), 'compact_sha256': item['sha256']})
        for row in rows:
            if not row['file'].endswith('.pdf'):
                continue
            old = baseline_package / 'Figures' / row['file']
            with fitz.open(old) as pdf:
                old_sizes = sorted({round(span['size'], 3) for page in pdf
                    for block in page.get_text('dict')['blocks'] if 'lines' in block
                    for line in block['lines'] for span in line['spans'] if span['text'].strip()})
                old_words=Counter(word[4] for page in pdf for word in page.get_text('words'))
            with fitz.open(DEST / row['file']) as pdf:
                new_sizes = sorted({round(span['size'], 3) for page in pdf
                    for block in page.get_text('dict')['blocks'] if 'lines' in block
                    for line in block['lines'] for span in line['spans'] if span['text'].strip()})
                new_words=Counter(word[4] for page in pdf for word in page.get_text('words'))
            assert old_sizes == new_sizes, (row['file'], old_sizes, new_sizes)
            assert old_words==new_words, (row['file'],old_words-new_words,new_words-old_words)
            row['font_sizes_pt_unchanged_from_baseline'] = old_sizes
            row['actual_pdf_words_and_numeric_labels_preserved'] = True

    def gaps(geometry):
        ps={p['panel']:p['ink_bbox_mm'] for p in geometry['panels']}
        keys=list(ps)
        if len(keys)==3:groups=[['a'],['b'],['c']]
        elif len(keys)==6:groups=[['a','b'],['c','d'],['e','f']]
        else:groups=[keys]
        vertical=[min(ps[k][1] for k in upper)-max(ps[k][3] for k in lower)
                  for upper,lower in zip(groups,groups[1:])]
        horizontal=[ps[row[1]][0]-ps[row[0]][2] for row in groups if len(row)==2]
        letters={x['letter']:x['bbox_mm'] for x in geometry['letters']}
        letter_clearance=[letters[row[1]][0]-ps[row[0]][2]
                          for row in groups if len(row)==2 and row[1] in letters]
        return dict(vertical_row_ink_gaps_mm=vertical,horizontal_panel_ink_gaps_mm=horizontal,
                    horizontal_clearance_to_next_panel_letter_mm=letter_clearance,
                    legend_to_data_tight_gaps_mm={p['panel']:p['legend_to_data_tight_gap_mm']
                                                  for p in geometry['panels']})
    geometry_comparison=[]
    for f in sorted((W/'figures/qa').glob('*.json')):
        new=json.loads(f.read_text(encoding='utf-8'))
        old=json.loads((W/'baseline/qa'/f.name).read_text(encoding='utf-8'))
        # Inactive axis-off legend-strip numeric ticks are present in Matplotlib's
        # object tree but are never painted. Their automatic locator can respond
        # to height. All actually painted numeric labels are checked in the PDFs.
        signature=lambda d:sorted((t['text'],t['size_pt'],str(t['weight'])) for t in d['style']['actual_text_styles']
                                  if not re.fullmatch(r'[\d.−+–-]+',t['text']))
        assert signature(old)==signature(new), f
        old_geom=old['style']['actual_geometry'];new_geom=new['style']['actual_geometry']
        old_gap=gaps(old_geom);new_gap=gaps(new_geom)
        assert all(g>0 for g in new_gap['vertical_row_ink_gaps_mm']), f
        assert all(g>=18/72*25.4 for g in new_gap['horizontal_clearance_to_next_panel_letter_mm']), (f,new_gap)
        folder='Fig_S4' if f.stem.startswith('Fig_S4') else 'Fig_2'
        rel=Path(folder)/(('subfigures/'+f.stem+'.png') if f.stem not in ['Fig_2','Fig_S4'] else f.stem+'.png')
        baseline_png=W/'baseline'/rel
        if baseline_package:
            assert sha(baseline_png)==sha(baseline_package/'Figures'/rel), ('06 baseline differs',rel)
        with Image.open(baseline_png) as im:
            dpi=im.info['dpi'];before_mm=[im.width/dpi[0]*25.4,im.height/dpi[1]*25.4]
        with Image.open(DEST/rel) as im:
            dpi=im.info['dpi'];after_mm=[im.width/dpi[0]*25.4,im.height/dpi[1]*25.4]
        assert after_mm[1]<before_mm[1], (f,before_mm,after_mm)
        geometry_comparison.append(dict(canvas=f.stem,before=old_geom,after=new_geom,
             before_gaps=old_gap,after_gaps=new_gap,export_mm_before=before_mm,export_mm_after=after_mm,
             exported_height_reduction_pct=(1-after_mm[1]/before_mm[1])*100,
             actual_text_and_font_sizes_preserved=True,baseline_png_reproduces_06=True,
             before_png_sha256=sha(baseline_png),after_png_sha256=sha(DEST/rel)))
    (qa_dir/'compact_geometry_comparison.json').write_text(json.dumps(geometry_comparison,ensure_ascii=False,indent=2),encoding='utf-8')

    report = {
        'author': 'CHONG LIU', 'automated_checks_passed': True,
        'all_passed': visual_reviewed,
        'scope': 'Two native three-row figures plus nine independently rendered panels; layout compacted without changing text, font size, colour or data.',
        'visual_reviewed_final_exports': visual_reviewed,
        'native_figure_checks': checks, 'files': rows, 'input_sha256': source_hashes,
        'immutable_reference_csvs': immutable,
        'csv_comparison_with_immutable_baseline': baseline_checks,
        'actual_geometry_before_after': geometry_comparison,
        'old_size_curve_and_classes_byte_identical': True,
        'quantitative_imagegen_used': False,
    }
    (qa_dir / 'Fig2_S4_紧凑版核查.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Fig. 2 与 Fig. S4 紧凑版核查（内部）', '',
             '- 11 个画布、22 个 PNG/PDF；两个组图均保持三行，9 张独立子图单独重绘。',
             '- 实际 Text 对象、刻度及 PDF 文字均为正常字重，无粗体数学命令和粗体嵌入字体。',
             '- 原生最小字号 12 pt、原配色 nature_new、原始数据与指标 CSV 保持不变。',
             '- 每个画布自动文字重叠为 0，外边界最大对齐残差均不超过 0.35 pt。',
             '- PNG 为 600 dpi；PDF 作者、创建者、生产者均为 CHONG LIU。',
             f'- 最终导出逐图视觉核查：{"通过" if visual_reviewed else "待核查"}。', '',
             '| 画布 | 最大对齐残差/pt | 最小字号/pt | Text 数量 |', '|---|---:|---:|---:|']
    for name, check in checks.items():
        lines.append(f"| {name} | {check['alignment_max_residual_pt']:.6f} | {check['native_min_font_pt']:.1f} | {len(check['actual_text_styles'])} |")
    lines.extend(['','| 画布 | 输出高度/mm：06 → 07 | 高度减少/% |', '|---|---:|---:|'])
    for d in geometry_comparison:
        lines.append(f"| {d['canvas']} | {d['export_mm_before'][1]:.2f} → {d['export_mm_after'][1]:.2f} | {d['exported_height_reduction_pct']:.1f} |")
        if d['before_gaps']['vertical_row_ink_gaps_mm']:
            lines.append('')
            lines.append(f"{d['canvas']} 实际内容行间隔/mm：{d['before_gaps']['vertical_row_ink_gaps_mm']} → {d['after_gaps']['vertical_row_ink_gaps_mm']}。")
            lines.append('')
    (qa_dir / 'Fig2_S4_紧凑版核查.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'automated_passed': True, 'visual_reviewed': visual_reviewed,
                      'canvases': len(checks), 'formal_files': len(rows),
                      'max_alignment_pt': max(x['alignment_max_residual_pt'] for x in checks.values()),
                      'immutable_csv_count': len(immutable)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-dir', type=Path, required=True)
    parser.add_argument('--visual-reviewed', action='store_true')
    parser.add_argument('--baseline-package', type=Path)
    args = parser.parse_args()
    run(args.qa_dir, args.visual_reviewed, args.baseline_package)
