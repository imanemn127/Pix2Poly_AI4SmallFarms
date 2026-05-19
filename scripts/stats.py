import json
import re
import sys
from collections import defaultdict
from statistics import mean, median

def process_file(path, label, write):
    with open(path) as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    n_images = len(images)
    n_anns = len(annotations)

    ann_per_patch = defaultdict(int)
    verts_per_patch = defaultdict(int)
    for img in images:
        ann_per_patch[img['id']] = 0
        verts_per_patch[img['id']] = 0

    total_verts = 0
    for ann in annotations:
        iid = ann['image_id']
        ann_per_patch[iid] += 1
        for seg in ann['segmentation']:
            v = len(seg) // 2
            verts_per_patch[iid] += v
            total_verts += v

    ac = list(ann_per_patch.values())
    vc = list(verts_per_patch.values())

    write(f'\n===== Patch size: {label}  |  {path} =====')
    write(f'Images (patches) : {n_images}')
    write(f'Total annotations: {n_anns}')
    write(f'Avg fields/patch : {n_anns/n_images:.1f}')
    write(f'Avg vertices/field: {total_verts/n_anns:.1f}' if n_anns else 'Avg vertices/field: 0')
    write(f'Avg vertices/patch: {total_verts/n_images:.0f}')
    write('')

    write('Per-patch annotation count:')
    write(f'  min={min(ac)}  max={max(ac)}  avg={mean(ac):.2f}  median={median(ac):.1f}')
    write('Per-patch vertex sum (all annotations):')
    write(f'  min={min(vc)}  max={max(vc)}  avg={mean(vc):.1f}  median={median(vc):.1f}')
    write('')

    write('Truncation analysis:')
    write('max_num_vertices  |  patches > max  (% of total)  |  vertices lost  (% of total)')
    write('------------------ | ---------------------------- | ------------------------------')
    for mv in [192, 256, 384, 512, 768, 1024]:
        over = sum(1 for v in vc if v > mv)
        lost = sum(v - mv for v in vc if v > mv)
        pct_img = over / n_images * 100
        pct_vert = lost / total_verts * 100 if total_verts > 0 else 0
        write(f'{mv:>18d} | {over:>6d}         ({pct_img:5.1f} %)   | {lost:>10d}   ({pct_vert:5.1f} %)')

    return {
        'label': label,
        'n_images': n_images, 'n_anns': n_anns,
        'ann_mean': mean(ac), 'ann_median': median(ac), 'ann_max': max(ac),
        'vert_mean': mean(vc), 'vert_median': median(vc), 'vert_max': max(vc),
    }

def extract_label(path, index):
    m = re.search(r'_(\d+)[/\\]', path)
    return int(m.group(1)) if m else f'File {index}'

def print_comparison_fixed(results, write):
    header = ("Patch Size (px)", "Tiles (images)", "Total Anns",
              "MeanAnns/tile", "MedAnns/tile", "MaxAnns/tile",
              "MeanVerts/tile", "MedVerts/tile", "MaxVerts/tile")
    widths = [14, 14, 12, 14, 12, 12, 14, 12, 12]  

    fmt = " | ".join(f"{{:>{w}}}" for w in widths)
    write(fmt.format(*header))
    write("-" * (sum(widths) + 3 * (len(widths)-1)))

    for r in results:
        write(fmt.format(
            str(r["label"]),
            str(r["n_images"]),
            str(r["n_anns"]),
            f"{r['ann_mean']:.2f}",
            f"{r['ann_median']:.1f}",
            str(r['ann_max']),
            f"{r['vert_mean']:.1f}",
            f"{r['vert_median']:.1f}",
            str(r['vert_max'])
        ))

def main():
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} output_coco_32/train_coco.json [output_coco_64/train_coco.json ...] [--out stats.txt]')
        sys.exit(1)

    # optional output filename at the end: --out filename
    args = sys.argv[1:]
    out_file = 'stats.txt'
    if '--out' in args:
        i = args.index('--out')
        if i + 1 < len(args):
            out_file = args[i + 1]
            # remove the two args so remaining are only input files
            args = args[:i] + args[i+2:]
        else:
            print('Error: --out requires a filename')
            sys.exit(1)

    results = []
    # open output file once
    with open(out_file, 'w', encoding='utf-8') as fout:
        def write(line=''):
            # write to stdout
            print(line)
            # write to file with newline
            fout.write(line + '\n')

        for i, path in enumerate(args, 1):
            label = extract_label(path, i)
            results.append(process_file(path, label, write))

        if len(results) < 2:
            write('\nNot enough files for comparison.')
            return

        write('\n\n===== Comparison across patch sizes =====')
        print_comparison_fixed(results, write)

    print(f'\nResults written to {out_file}')

if __name__ == '__main__':
    main()
