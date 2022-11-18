import numpy as np

n_cmp = [5, 10, 15, 20, 30, 50, 100, 300]


def Recall(top_k, origin, n):
    m = {}
    for i in n_cmp:
        m[i] = __recall(top_k, origin, i)
        if i == n:
            break
    return m


def __recall(top_k, origin, n):
    rc_sum, rc = 0, 0
    for i, q in enumerate(origin):
        # 排除掉 mask_zero
        s1 = {a for a in q if a != 0}
        avail = min(np.sum(np.array(top_k[i]) != -1), n)
        rc_sum += len(s1)
        rc += len(s1.intersection(top_k[i][:avail]))
    return rc / rc_sum


def HitRate(top_k, origin, n):
    m = {}
    for i in n_cmp:
        m[i] = __hit_rate(top_k, origin, i)
        if i == n:
            break
    return m


def __hit_rate(top_k, origin, n):
    rc_sum, rc = 0, 0
    for i, q in enumerate(origin):
        # 排除掉 mask_zero
        s1 = {a for a in q if a != 0}
        avail = min(np.sum(np.array(top_k[i]) != -1), n)
        rc_sum += avail
        rc += len(s1.intersection(top_k[i][:n]))
    return rc / rc_sum

    # batch_size = test_size
    # r5, r10, r30, r50, r100, rc_cnt, hr_cnt = 0, 0, 0, 0, 0, 0, 0
    # for i, q in enumerate(test_user_data[topk_cmp_col]):
    #     s1 = {a for a in q if a != 0}
    #     rc_cnt += len(s1)
    #     hr_cnt += np.sum(np.array(top_k[i]) != -1)
    #     r5 += len(s1.intersection(top_k[i][:5]))
    #     r10 += len(s1.intersection(top_k[i][:10]))
    #     r30 += len(s1.intersection(top_k[i][:30]))
    #     r50 += len(s1.intersection(top_k[i][:50]))
    #     r100 += len(s1.intersection(top_k[i][:100]))
    # info = f'Recall@5: {r5 / rc_cnt:.4f}, Recall@10: {r10 / rc_cnt:.4f},' \
    #        f'Recall@30: {r30 / rc_cnt:.4f}, Recall@50: {r50 / rc_cnt:.4f}, Recall@100: {r100 / rc_cnt:.4f}\n'
    # info += f'HR@5: {r5 / min(hr_cnt, 5 * batch_size):.4f}, HR@10: {r10 / min(hr_cnt, 10 * batch_size):.4f}, ' \
    #         f'HR@30: {r30 / min(hr_cnt, 30 * batch_size):.4f}, HR@50: {r50 / min(hr_cnt, 50 * batch_size):.4f}, ' \
    #         f'HR@100: {r100 / min(hr_cnt, 50 * batch_size):.4f}\n'
