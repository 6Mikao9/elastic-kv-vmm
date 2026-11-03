"""Compare the Triton resident lookup with independent CPU lookup vectors."""
import torch

from adaptive_sparse.resident_lookup import resolve_resident


def main():
    topk = [[1, -1, 2] + [-1] * 13, [0] * 16, [0, 2, -1] + [-1] * 13]
    rows = [3, 0, 1]
    lengths = [130, 0, 0]
    table = [[(i * 300 + j) for j in range(300)] for i in range(4)]
    mapping = [i * 3 + 5 for i in range(1200)]
    selected = [2, 0]
    expected = [[-77] * 2048 for _ in rows]
    for row in selected:
        expected[row] = [
            mapping[table[rows[row]][block * 128 + j]]
            if block >= 0 and block * 128 + j < lengths[row] else -1
            for block in topk[row] for j in range(128)
        ]
    assert expected[0][:3] == [3089, 3092, -1]
    assert expected[0][-1] == -1 and expected[2] == [-1] * 2048
    assert expected[1] == [-77] * 2048
    if not torch.cuda.is_available():
        print("CPU reference passed")
        return

    def strided(values, dtype):
        source = torch.tensor(values, dtype=dtype, device="cuda")
        backing = torch.empty((*source.shape, 2), dtype=dtype, device="cuda")
        view = backing[..., 1]
        view.copy_(source)
        return view

    output = torch.full((1, 3, 4096), -77, device="cuda", dtype=torch.int32)[..., ::2]
    resolve_resident(strided([topk], torch.int32), strided(table, torch.long),
                     strided(mapping, torch.long), strided(rows, torch.long),
                     strided(lengths, torch.int32), strided(selected, torch.long),
                     output, 128)
    assert output.tolist() == [expected]
    print("CUDA resident lookup passed")


if __name__ == "__main__":
    main()
