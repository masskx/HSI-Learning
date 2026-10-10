// Build the 12 HSI-Learning classroom decks with pptxgenjs (portable: Node only).
//
//   python scripts/export_slide_content.py --output results/slide_revision_build/content.json
//   cd slides && npm install            # once
//   node scripts/build_course_slides.js results/slide_revision_build/content.json [L01 L05A ...]
//
// `python scripts/build_course_slides.py` runs both steps. Content comes only from
// slides/src/course_content.py + classroom_content.py (exported to content.json); this
// file is the design system. Same pages and speaker notes as the previous artifact-tool
// build (build_course_slides.mjs), new visual design:
//   - "sandwich": midnight cover and closing slide, white content slides;
//   - one motif: spectral chips -- numbered markers coloured along the visible spectrum,
//     plus the real Indian Pines mean spectrum as the cover graphic (same as the video intro);
//   - Microsoft YaHei for latin AND East-Asian runs (theme + every run), Consolas for code;
//   - every figure is fitted to its real aspect ratio: tall figures sit beside a reading
//     guide, wide figures span the slide with the guide underneath.
'use strict';
const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..');
const NODE_MODULES = path.join(ROOT, 'slides', 'node_modules');
const pptxgen = require(require.resolve('pptxgenjs', { paths: [NODE_MODULES, __dirname] }));
const JSZip = require(require.resolve('jszip', { paths: [NODE_MODULES, __dirname] }));

// ------------------------------------------------------------------ design tokens
const FONT = 'Microsoft YaHei';
const MONO = 'Consolas';
const THEME = {
  name: 'HSI-Learning Spectral',
  colors: {
    dk1: '14233A',     // ink: titles and body
    lt1: 'FFFFFF',
    dk2: '0D1B2E',     // midnight: cover, closing, code panels
    lt2: 'F2F5F9',     // quiet panel
    accent1: '2F5BB7', // deep blue: structure, labels
    accent2: '1F8FB8', // spectral teal
    accent3: '2E9E4F', // spectral green
    accent4: 'E8752A', // spectral orange
    accent5: '5B3FA0', // spectral violet
    accent6: 'D33A2C', // spectral red
    hlink: '2F5BB7',
    folHlink: '5B3FA0',
  },
};
const X = THEME.colors;
const SUB = '5F6E80';        // secondary text on white
const LINE = 'DCE3EA';       // hairlines, table rules
const MIST = 'A9BACD';       // secondary text on midnight
const CARD_DARK = '16294A';  // card on midnight
const CODE_FG = 'D7E3F0';
const AMBER = 'F5B54A';      // run output on dark panels
// Spectral chip sequence, violet -> red, sampled evenly for n items.
const SPECTRUM = [X.accent5, X.accent1, X.accent2, X.accent3, X.accent4, X.accent6];
const chipColor = (i, n) => SPECTRUM[n <= 1 ? 0 : Math.round((i * (SPECTRUM.length - 1)) / Math.max(n - 1, 1))];

const W = 13.333, H = 7.5, M = 0.75, CW = W - 2 * M;
const COVERS = { L00: '环境与\n第一次前向', L01: '理解\n高光谱数据', L02: '划分与\n评价指标', L03: '建立可信的\nSVM 基线', L04: '从像元\n构建 Patch', L05A: '模型的一次\n参数更新', L05: '卷积的轴\n与感受野', L06: '读懂\nHybridSN', L07: '类别不均衡\n与错误分析', L08: '原型网络\n与少样本任务', L09: '未知类\n与拒识决策', L10: '从实验记录\n到研究证据' };
const BRAND = path.join(ROOT, 'slides', 'assets', 'brand', 'spectrum-bars.png');

// ------------------------------------------------------------------ helpers
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fit(file, box) {
  const { w, h } = pngSize(file);
  const s = Math.min(box.w / w, box.h / h);
  const fw = w * s, fh = h * s;
  return { x: box.x + (box.w - fw) / 2, y: box.y + (box.h - fh) / 2, w: fw, h: fh };
}
function T(slide, text, o) {
  slide.addText(text, {
    isTextBox: true, margin: 0, fontFace: o.mono ? MONO : FONT, fontSize: 18, color: X.dk1,
    valign: 'top', ...o, mono: undefined,
  });
}
function rect(slide, pres, o) {
  slide.addShape(o.radius ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, {
    x: o.x, y: o.y, w: o.w, h: o.h, fill: o.fill ? { color: o.fill } : { type: 'none' },
    line: o.line ? { color: o.line, width: o.lineW || 0.75 } : { type: 'none' },
    rectRadius: o.radius, objectName: o.name,
  });
}
function chip(slide, pres, x, y, label, color, size = 0.46) {
  rect(slide, pres, { x, y, w: size, h: size, fill: color, radius: 0.09, name: 'chip ' + label });
  T(slide, label, { x, y, w: size, h: size, fontSize: size > 0.5 ? 16 : 13, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
}
// Numbered points with spectral chips, distributed over a vertical band.
function points(slide, pres, items, box, size = 20) {
  const n = items.length, gap = box.h / n;
  items.forEach((text, i) => {
    const y = box.y + i * gap;
    chip(slide, pres, box.x, y + 0.04, String(i + 1).padStart(2, '0'), chipColor(i, n));
    T(slide, text, { x: box.x + 0.68, y, w: box.w - 0.68, h: Math.max(gap - 0.12, 0.55), fontSize: size, lineSpacingMultiple: 1.15 });
  });
}
function runPanel(slide, pres, run, y, h = 1.08) {
  rect(slide, pres, { x: M, y, w: CW, h, fill: X.dk2, radius: 0.1, name: 'run panel' });
  const long = Math.max(run[0].length, run[1].length) > 70;
  T(slide, [{ text: '>>> ', options: { color: AMBER, bold: true } }, { text: run[0], options: { color: CODE_FG } }],
    { x: M + 0.35, y: y + 0.17, w: CW - 0.7, h: 0.36, mono: true, fontSize: long ? 14 : 16 });
  T(slide, run[1], { x: M + 0.35, y: y + h - 0.5, w: CW - 0.7, h: 0.36, mono: true, fontSize: long ? 14 : 16, bold: true, color: AMBER });
}
function label(slide, text, x, y, w, color = X.accent1) {
  T(slide, text, { x, y, w, h: 0.32, fontSize: 13, bold: true, color, charSpacing: 1 });
}
function framePicture(slide, pres, file, box, fill = X.lt2) {
  rect(slide, pres, { ...box, fill, radius: 0.08, name: 'figure frame' });
  const pad = 0.18;
  const f = fit(file, { x: box.x + pad, y: box.y + pad, w: box.w - 2 * pad, h: box.h - 2 * pad });
  slide.addImage({ path: file, ...f, altText: path.basename(file) });
}
// Vertical flow of steps: numbered cards joined by arrows.
function flowV(slide, pres, labels, box) {
  const n = labels.length, gap = 0.28, ch = Math.min(0.92, (box.h - gap * (n - 1)) / n);
  const total = n * ch + (n - 1) * gap, y0 = box.y + (box.h - total) / 2;
  labels.forEach((v, i) => {
    const y = y0 + i * (ch + gap);
    rect(slide, pres, { x: box.x, y, w: box.w, h: ch, fill: X.lt2, radius: 0.08, name: 'step ' + (i + 1) });
    chip(slide, pres, box.x + 0.22, y + (ch - 0.46) / 2, String(i + 1).padStart(2, '0'), chipColor(i, n));
    T(slide, v, { x: box.x + 0.88, y, w: box.w - 1.05, h: ch, fontSize: 18, bold: true, valign: 'middle' });
    if (i < n - 1) slide.addShape(pres.shapes.LINE, { x: box.x + 0.45, y: y + ch, w: 0, h: gap, line: { color: SUB, width: 1.25, endArrowType: 'triangle' } });
  });
}
// Horizontal flow, used under a formula card.
function flowH(slide, pres, labels, box) {
  const n = labels.length, gap = 0.34, cw = (box.w - gap * (n - 1)) / n;
  labels.forEach((v, i) => {
    const x = box.x + i * (cw + gap);
    rect(slide, pres, { x, y: box.y, w: cw, h: box.h, fill: X.lt2, radius: 0.08, name: 'step ' + (i + 1) });
    chip(slide, pres, x + 0.18, box.y + 0.18, String(i + 1).padStart(2, '0'), chipColor(i, n), 0.38);
    T(slide, v, { x: x + 0.18, y: box.y + 0.66, w: cw - 0.36, h: box.h - 0.8, fontSize: 15, bold: true });
    if (i < n - 1) slide.addShape(pres.shapes.LINE, { x: x + cw + 0.05, y: box.y + box.h / 2, w: gap - 0.1, h: 0, line: { color: SUB, width: 1.25, endArrowType: 'triangle' } });
  });
}

// ------------------------------------------------------------------ layouts
function defineLayouts(pres, lesson) {
  const header = (dark) => [
    { rect: { x: M, y: 0.42, w: 0.72, h: 0.3, fill: { color: dark ? X.accent1 : X.dk2 }, rectRadius: 0.06 } },
    { text: { text: lesson.id, options: { x: M, y: 0.42, w: 0.72, h: 0.3, margin: 0, align: 'center', valign: 'middle', fontFace: FONT, fontSize: 11, bold: true, color: 'FFFFFF' } } },
    { text: { text: 'HSI-LEARNING  ·  高光谱图像分类课程', options: { x: M + 0.9, y: 0.42, w: 6, h: 0.3, margin: 0, valign: 'middle', fontFace: FONT, fontSize: 11, color: dark ? MIST : SUB, charSpacing: 1.5 } } },
  ];
  const title = (color) => ({ placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.95, w: CW, h: 0.8, margin: 0, valign: 'middle', fontFace: FONT, fontSize: 30, bold: true, color, align: 'left' }, text: '' } });
  const number = (color) => ({ x: W - M - 0.9, y: 0.42, w: 0.9, h: 0.3, fontFace: FONT, fontSize: 11, color, align: 'right', valign: 'middle', margin: 0 });
  pres.defineSlideMaster({ title: 'COVER', background: { color: X.dk2 }, objects: [
    { image: { path: BRAND, x: 0, y: H - 1.55, w: W, h: 1.55, transparency: 15 } },
    { text: { text: 'HSI-LEARNING  ·  高光谱图像分类课程', options: { x: M, y: 0.62, w: 8, h: 0.32, margin: 0, fontFace: FONT, fontSize: 12, bold: true, color: MIST, charSpacing: 3 } } },
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 2.0, w: 6.9, h: 1.95, margin: 0, valign: 'top', fontFace: FONT, fontSize: 50, bold: true, color: 'FFFFFF', align: 'left', lineSpacingMultiple: 1.08 }, text: '' } },
  ] });
  pres.defineSlideMaster({ title: 'CONTENT', background: { color: X.lt1 }, objects: [...header(false), title(X.dk1)], slideNumber: number(SUB) });
  pres.defineSlideMaster({ title: 'CLOSING', background: { color: X.dk2 }, objects: [
    { image: { path: BRAND, x: 0, y: H - 1.05, w: W, h: 1.05, transparency: 45 } },
    ...header(true), title('FFFFFF')], slideNumber: number(MIST) });
}

// ------------------------------------------------------------------ slide types
function notesFrame(data, l, notes) {
  return notes + `\n课堂入口: ${l.notebook || 'docs/teaching/research-case.md'}\n定位: ${l.cells.join(', ')}\n先预测，再运行；答错时回到手算例。\nAI辅助：${l.ai}\n人工核验：${l.verify}\n`
    + l.sources.map((k) => (data.sources[k] || []).join('\n')).join('\n') + '\n运行配置、数据哈希和结果：artifacts/teaching/classroom-facts.json';
}

function cover(pres, data, l) {
  const s = pres.addSlide({ masterName: 'COVER', sectionTitle: '开场' });
  s.addText(COVERS[l.id] || l.title, { placeholder: 'title' });
  rect(s, pres, { x: M, y: 1.36, w: 0.86, h: 0.38, fill: X.accent1, radius: 0.07, name: 'lesson id' });
  T(s, l.id, { x: M, y: 1.36, w: 0.86, h: 0.38, fontSize: 15, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
  T(s, [{ text: '本课问题  ', options: { color: AMBER, bold: true } }, { text: l.title, options: { color: 'FFFFFF' } }],
    { x: M, y: 4.08, w: 7.0, h: 0.42, fontSize: 19 });
  T(s, l.subtitle || '', { x: M, y: 4.6, w: 7.0, h: 0.36, fontSize: 16, color: MIST });
  T(s, '先修  ·  ' + l.prerequisites, { x: M, y: 5.08, w: 7.0, h: 0.34, fontSize: 13, color: MIST });
  // Same hero rule as before (first figure of the lesson, else the L01 spectra), but prefer a
  // figure that reads at cover size and shape the card to the figure's aspect ratio.
  const ratio = (e) => { const d = pngSize(e.image_asset); return d.w / d.h; };
  const figures = l.pages.filter((e) => e.image_asset);
  const hero = figures.find((e) => ratio(e) < 1.9) || figures[0]
    || data.lessons.find((e) => e.id === 'L01').pages.find((e) => e.title.includes('光谱') && e.image_asset);
  if (hero) {
    const r = ratio(hero), cw = r >= 1.9 ? 5.1 : 4.48;
    const ch = Math.min(3.95, Math.max(2.2, (cw - 0.44) / r + 0.44));
    const box = { x: W - M - cw, y: 1.3 + (3.95 - ch) / 2, w: cw, h: ch };
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { ...box, fill: { color: 'FFFFFF' }, line: { type: 'none' }, rectRadius: 0.1, objectName: 'hero frame',
      shadow: { type: 'outer', color: '000000', opacity: 0.35, blur: 18, offset: 6, angle: 90 } });
    const f = fit(hero.image_asset, { x: box.x + 0.22, y: box.y + 0.22, w: box.w - 0.44, h: box.h - 0.44 });
    s.addImage({ path: hero.image_asset, ...f, altText: 'cover figure' });
    // Recorded for the video intro, which ends on this exact cover layout.
    var heroInfo = { image: path.relative(ROOT, hero.image_asset).split(path.sep).join('/'), card: box, figure: f };
  }
  s.addNotes(notesFrame(data, l, `开场问题：${l.exit_question}\n原标题：${l.title}\n参考节奏：5分钟回顾，10分钟手算，10分钟代码，12分钟练习，5分钟错误分析，3分钟退出题。`)
    + '\n封面图源：' + (hero?.image?.join(' / ') || ''));
  return { type: 'cover', title: l.title, cover_title: COVERS[l.id] || l.title, hero: heroInfo || null };
}

function objectives(pres, data, l) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '开场' });
  s.addText('这节课，你将完成什么', { placeholder: 'title' });
  const n = l.outputs.length, gap = 0.3, cw = (CW - gap * (n - 1)) / n;
  l.outputs.forEach((v, i) => {
    const x = M + i * (cw + gap);
    rect(s, pres, { x, y: 2.1, w: cw, h: 2.45, fill: X.lt2, radius: 0.1, name: 'deliverable ' + (i + 1) });
    chip(s, pres, x + 0.35, 2.45, String(i + 1).padStart(2, '0'), chipColor(i, n), 0.56);
    T(s, v, { x: x + 0.35, y: 3.25, w: cw - 0.7, h: 1.1, fontSize: 22, bold: true, lineSpacingMultiple: 1.15 });
  });
  const info = [['NOTEBOOK', l.notebook || 'docs/teaching/research-case.md', true], ['定位 CELL', l.cells.join(' · '), true], ['作业', `${l.assignment} · 完整示例 / 填空 / 迁移`, false]];
  const iw = (CW - 0.6) / 3;
  s.addShape(pres.shapes.LINE, { x: M, y: 5.3, w: CW, h: 0, line: { color: LINE, width: 1 } });
  info.forEach(([k, v, mono], i) => {
    const x = M + i * (iw + 0.3);
    label(s, k, x, 5.55, iw, SUB);
    T(s, v, { x, y: 5.95, w: iw, h: 0.8, fontSize: mono ? 13 : 15, mono, lineSpacingMultiple: 1.2 });
  });
  s.addNotes(notesFrame(data, l, '交付物以代码、配置、输出和解释为准。先做完整示例，再做填空与迁移任务。'));
  return { type: 'objectives', title: '这节课，你将完成什么' };
}

function contentPage(pres, data, l, e) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '讲解' });
  s.addText(e.title, { placeholder: 'title' });
  const CT = 2.05, bottom = e.run ? 5.5 : 6.95;
  if (e.image_asset) {
    const { w, h } = pngSize(e.image_asset);
    const src = e.image ? `${e.image[0]}  ·  cell ${e.image[1]}` : '';
    if (w / h >= 1.9) {
      const fh = Math.min(3.55, (CW - 0.36) / (w / h) + 0.36);
      framePicture(s, pres, e.image_asset, { x: M, y: CT, w: CW, h: fh });
      const n = e.points.length, gap = 0.35, cw = (CW - gap * (n - 1)) / n, py = CT + fh + 0.35;
      e.points.forEach((v, i) => {
        const x = M + i * (cw + gap);
        chip(s, pres, x, py + 0.04, String(i + 1).padStart(2, '0'), chipColor(i, n), 0.4);
        T(s, v, { x: x + 0.56, y: py, w: cw - 0.56, h: 0.85, fontSize: 16, lineSpacingMultiple: 1.15 });
      });
      T(s, src, { x: M, y: 7.0, w: CW, h: 0.25, fontSize: 9, color: SUB, mono: true });
    } else {
      framePicture(s, pres, e.image_asset, { x: M, y: CT, w: 7.35, h: 4.9 });
      label(s, '怎么读这张图', 8.5, CT + 0.05, 4.08);
      points(s, pres, e.points, { x: 8.5, y: CT + 0.6, w: 4.08, h: 3.6 }, 17);
      T(s, src, { x: 8.5, y: 6.45, w: 4.08, h: 0.5, fontSize: 9, color: SUB, mono: true });
    }
  } else if (e.table) {
    const lead = e.points.join('   ·   '), ty = lead.length > 52 ? 3.1 : 2.85;
    T(s, lead, { x: M, y: CT, w: CW, h: ty - CT - 0.2, fontSize: 17, color: SUB, lineSpacingMultiple: 1.2 });
    const rows = e.table.map((row, r) => row.map((v) => ({ text: v, options: r === 0
      ? { bold: true, color: 'FFFFFF', fill: { color: X.dk2 } }
      : { color: X.dk1, fill: { color: r % 2 ? 'FFFFFF' : X.lt2 } } })));
    const rh = Math.min(0.8, (6.85 - ty) / rows.length);
    s.addTable(rows, { x: M, y: ty, w: CW, colW: Array(rows[0].length).fill(CW / rows[0].length), rowH: rh,
      fontFace: FONT, fontSize: rows.length > 5 ? 16 : 18, valign: 'middle', margin: [0, 0.18, 0, 0.18],
      border: { type: 'solid', pt: 0.75, color: LINE } });
  } else if (e.formula_asset) {
    points(s, pres, e.points, { x: M, y: CT + 0.1, w: 5.35, h: bottom - CT - 0.2 }, 19);
    const cardH = e.diagram ? 1.75 : bottom - CT - 0.15;
    rect(s, pres, { x: 6.55, y: CT, w: 6.03, h: cardH, fill: X.lt2, radius: 0.1, name: 'formula card' });
    const { w, h } = pngSize(e.formula_asset);
    const natural = { w: (w / 300) * 1.7, h: (h / 300) * 1.7 };
    const box = { w: Math.min(natural.w, 5.4), h: Math.min(natural.h, cardH - 0.4) };
    const f = fit(e.formula_asset, { x: 6.55 + (6.03 - box.w) / 2, y: CT + (cardH - box.h) / 2, w: box.w, h: box.h });
    s.addImage({ path: e.formula_asset, ...f, altText: 'formula: ' + (e.math || '') });
    if (e.diagram) flowH(s, pres, e.diagram.labels, { x: 6.55, y: CT + 1.95, w: 6.03, h: bottom - CT - 2.05 });
  } else if (e.diagram) {
    points(s, pres, e.points, { x: M, y: CT + 0.1, w: 5.6, h: bottom - CT - 0.2 }, 20);
    flowV(s, pres, e.diagram.labels, { x: 7.0, y: CT, w: 5.58, h: bottom - CT });
  } else if (e.visual && e.visual.length) {
    points(s, pres, e.points, { x: M, y: CT + 0.1, w: 7.0, h: bottom - CT - 0.2 }, 21);
    rect(s, pres, { x: 8.35, y: CT, w: 4.23, h: bottom - CT, fill: X.lt2, radius: 0.1, name: 'keywords' });
    label(s, '关键词', 8.7, CT + 0.3, 3.5);
    const n = e.visual.length, step = Math.min(0.85, (bottom - CT - 1.3) / Math.max(n - 1, 1));
    const y0 = CT + 0.75 + (bottom - CT - 0.75 - step * (n - 1)) / 2;
    if (n > 1) s.addShape(pres.shapes.LINE, { x: 8.86, y: y0, w: 0, h: step * (n - 1), line: { color: LINE, width: 1.5 } });
    e.visual.forEach((v, i) => {
      const y = y0 + i * step;
      s.addShape(pres.shapes.OVAL, { x: 8.76, y: y - 0.1, w: 0.2, h: 0.2, fill: { color: chipColor(i, n) }, line: { color: X.lt2, width: 2 } });
      T(s, v, { x: 9.2, y: y - 0.2, w: 3.2, h: 0.4, fontSize: 16, bold: true, valign: 'middle' });
    });
  } else {
    const n = e.points.length, gap = 0.3, cw = (CW - gap * (n - 1)) / n, ch = e.run ? 3.1 : 2.9;
    e.points.forEach((v, i) => {
      const x = M + i * (cw + gap);
      rect(s, pres, { x, y: CT + 0.1, w: cw, h: ch, fill: X.lt2, radius: 0.1, name: 'point ' + (i + 1) });
      chip(s, pres, x + 0.35, CT + 0.45, String(i + 1).padStart(2, '0'), chipColor(i, n), 0.52);
      T(s, v, { x: x + 0.35, y: CT + 1.25, w: cw - 0.7, h: ch - 1.35, fontSize: n > 3 ? 18 : 20, bold: true, lineSpacingMultiple: 1.2 });
    });
  }
  if (e.run) runPanel(s, pres, e.run, 5.78);
  s.addNotes(notesFrame(data, l, (e.notes || '') + '\n' + (e.visual || []).join(' / ') + (e.image ? '\n图源：' + e.image.join(' / ') : '')));
  return { type: e.table ? 'table' : e.diagram ? 'diagram' : e.image ? 'image' : 'theory', title: e.title, image: e.image || null, math: e.math || null };
}

function practice(pres, data, l) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '练习' });
  s.addText('让代码验证你的判断', { placeholder: 'title' });
  const lines = l.code.split('\n'), longest = Math.max(...lines.map((v) => v.length));
  const size = longest <= 50 ? 16 : longest <= 58 ? 15 : 14;
  rect(s, pres, { x: M, y: 2.05, w: 8.1, h: 4.1, fill: X.dk2, radius: 0.1, name: 'code panel' });
  ['D33A2C', 'F2C316', '2E9E4F'].forEach((c, i) => s.addShape(pres.shapes.OVAL, { x: M + 0.3 + i * 0.24, y: 2.27, w: 0.13, h: 0.13, fill: { color: c }, line: { type: 'none' } }));
  T(s, lines.map((v, i) => {
    const [code, ...rest] = v.split('#');
    const runs = [{ text: code, options: { color: CODE_FG } }];
    if (rest.length) runs.push({ text: '#' + rest.join('#'), options: { color: '7E99AD' } });
    runs[runs.length - 1].options.breakLine = i < lines.length - 1;
    return runs;
  }).flat(), { x: M + 0.4, y: 2.65, w: 7.3, h: 3.3, mono: true, fontSize: size, lineSpacingMultiple: 1.45 });
  label(s, '打开', 9.25, 2.1, 3.33, SUB);
  T(s, l.notebook || 'docs/teaching/research-case.md', { x: 9.25, y: 2.45, w: 3.33, h: 0.75, mono: true, fontSize: 13, lineSpacingMultiple: 1.15 });
  label(s, '定位 cell', 9.25, 3.4, 3.33, SUB);
  l.cells.slice(0, 4).forEach((c, i) => {
    rect(s, pres, { x: 9.25, y: 3.8 + i * 0.52, w: 3.33, h: 0.4, line: LINE, radius: 0.06, name: 'cell ' + c });
    T(s, c, { x: 9.4, y: 3.8 + i * 0.52, w: 3.1, h: 0.4, mono: true, fontSize: 12, valign: 'middle', color: X.accent1 });
  });
  T(s, '先预测输出，再运行对应单元格。', { x: M, y: 6.4, w: CW, h: 0.4, fontSize: 17, color: SUB });
  s.addNotes(notesFrame(data, l, '只运行指定短cell；完整训练已在课前完成。缺资产时使用已执行输出备用并说明。'));
  return { type: 'practice', title: '让代码验证你的判断' };
}

function exercise(pres, data, l) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '练习' });
  s.addText('停下来，做一次判断', { placeholder: 'title' });
  rect(s, pres, { x: M, y: 2.05, w: 6.95, h: 4.75, fill: X.lt2, radius: 0.1, name: 'prediction card' });
  label(s, '课堂预测', M + 0.45, 2.45, 6, X.accent1);
  T(s, l.change, { x: M + 0.45, y: 2.95, w: 6.05, h: 2.6, fontSize: 24, lineSpacingMultiple: 1.3 });
  T(s, `作业 ${l.assignment}  ·  完整示例 / 填空 / 迁移`, { x: M + 0.45, y: 6.05, w: 6.05, h: 0.4, fontSize: 14, color: SUB });
  rect(s, pres, { x: 8.0, y: 2.05, w: 4.58, h: 4.75, fill: X.dk2, radius: 0.1, name: 'exit question card' });
  label(s, '退出题', 8.45, 2.45, 3.7, AMBER);
  T(s, l.exit_question, { x: 8.45, y: 2.95, w: 3.7, h: 3.4, fontSize: 24, bold: true, color: 'FFFFFF', lineSpacingMultiple: 1.3 });
  s.addNotes(notesFrame(data, l, '停下来收集预测，发现错误后重新解释；不等到课后作业才知道学生没理解。\n作业：' + l.assignment + '；完整示例 / 填空 / 迁移；提交代码、配置、结果和限制。'));
  return { type: 'exercise', title: '停下来，做一次判断' };
}

function limits(pres, data, l) {
  const s = pres.addSlide({ masterName: 'CLOSING', sectionTitle: '收束' });
  s.addText('结论要有边界', { placeholder: 'title' });
  const n = l.limits.length, gap = 0.3, cw = (CW - gap * (n - 1)) / n;
  l.limits.forEach((v, i) => {
    const x = M + i * (cw + gap);
    rect(s, pres, { x, y: 2.1, w: cw, h: 3.0, fill: CARD_DARK, radius: 0.1, name: i === 0 ? 'known' : 'boundary' });
    rect(s, pres, { x: x + 0.4, y: 2.5, w: 0.86, h: 0.38, fill: i === 0 ? X.accent2 : X.accent4, radius: 0.07, name: 'tag' });
    T(s, i === 0 ? '已知' : '边界', { x: x + 0.4, y: 2.5, w: 0.86, h: 0.38, fontSize: 14, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
    T(s, v, { x: x + 0.4, y: 3.2, w: cw - 0.8, h: 1.9, fontSize: 24, bold: true, color: 'FFFFFF', lineSpacingMultiple: 1.25 });
  });
  T(s, '来源、复现入口与辅助提示见讲者备注。', { x: M, y: 5.72, w: CW, h: 0.32, fontSize: 13, color: MIST });
  s.addNotes(notesFrame(data, l, '区分已观察事实、候选解释与下一项检验。\n退出题：' + l.exit_question));
  return { type: 'limits', title: '结论要有边界' };
}

// ------------------------------------------------------------------ theme
// pptxgenjs writes Office's palette and only latin theme fonts; write the course palette
// and set the East-Asian theme fonts so Chinese never falls back to a default typeface.
async function applyTheme(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const name = 'ppt/theme/theme1.xml';
  let xml = await zip.file(name).async('string');
  for (const [slot, hex] of Object.entries(THEME.colors)) {
    const re = new RegExp(`<a:${slot}>[\\s\\S]*?</a:${slot}>`);
    if (!re.test(xml)) throw new Error('theme slot missing: ' + slot);
    xml = xml.replace(re, `<a:${slot}><a:srgbClr val="${hex}"/></a:${slot}>`);
  }
  xml = xml.replace(/<a:clrScheme name="[^"]*"/, `<a:clrScheme name="${THEME.name}"`)
    .replace(/<a:ea typeface="[^"]*"\s*\/>/g, `<a:ea typeface="${FONT}"/>`)
    .replace(/<a:font script="Hans" typeface="[^"]*"\s*\/>/g, `<a:font script="Hans" typeface="${FONT}"/>`);
  zip.file(name, xml);
  // Code runs: Consolas for latin, YaHei for Chinese comments (Consolas has no CJK glyphs).
  for (const part of Object.keys(zip.files).filter((f) => /^ppt\/(slides|slideLayouts)\/[^/]+\.xml$/.test(f))) {
    const body = await zip.file(part).async('string');
    zip.file(part, body.replace(/<a:ea typeface="Consolas"/g, `<a:ea typeface="${FONT}"`));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
}

// ------------------------------------------------------------------ main
async function main() {
  const contentPath = process.argv[2] || path.join(ROOT, 'results/slide_revision_build/content.json');
  const data = JSON.parse(fs.readFileSync(contentPath, 'utf8'));
  const selected = process.argv.slice(3);
  const outDir = path.join(ROOT, 'slides', 'decks');
  fs.mkdirSync(outDir, { recursive: true });
  const manifest = [];
  for (const l of data.lessons) {
    if (selected.length && !selected.includes(l.id)) continue;
    const pres = new pptxgen();
    pres.layout = 'LAYOUT_WIDE';
    pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
    pres.title = `${l.id} ${l.title}`;
    pres.author = 'HSI-Learning';
    defineLayouts(pres, l);
    ['开场', '讲解', '练习', '收束'].forEach((title) => pres.addSection({ title }));
    const pages = [cover(pres, data, l), objectives(pres, data, l)];
    for (const e of l.pages) pages.push(contentPage(pres, data, l, e));
    pages.push(practice(pres, data, l), exercise(pres, data, l), limits(pres, data, l));
    const name = `${l.id}-${l.slug}.pptx`;
    const file = path.join(outDir, name);
    await pres.writeFile({ fileName: file });
    await applyTheme(file);
    manifest.push({ id: l.id, title: l.title, path: 'decks/' + name, slides: pages.length, pages, notebook: l.notebook, cells: l.cells, generated: true, rendered: false, visual_review: 'pending' });
    console.log(`Built ${name}: ${pages.length} slides`);
  }
  const manifestPath = path.join(ROOT, 'slides/lesson-manifest.json');
  let decks = manifest;
  if (selected.length && fs.existsSync(manifestPath)) {
    const previous = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const updated = new Map([...previous.decks, ...manifest].map((d) => [d.id, d]));
    decks = data.lessons.map((l) => updated.get(l.id)).filter(Boolean);
  }
  fs.writeFileSync(manifestPath, JSON.stringify({
    design: { engine: 'pptxgenjs', builder: 'scripts/build_course_slides.js', style: '光谱标识：午夜蓝封面与收尾、白底内容页、光谱色编号、真实平均光谱封面图', version: 3,
      fonts: { cjk: FONT, code: MONO }, source: 'slides/src/course_content.py + classroom_content.py' },
    decks, note: 'Package validation is separate from visual inspection and actual trial teaching.' }, null, 2));
}
main().catch((err) => { console.error(err); process.exit(1); });
