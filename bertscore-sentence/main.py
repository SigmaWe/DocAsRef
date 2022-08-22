from dataloader import newsroom
from eval import score


def newsroom_eval():
    print('[Newsroom]')
    sys_summaries, ref_summaries, docs, _ = newsroom.read('dataloader')
    results = score(sys_summaries, docs)
    print(results)


if __name__ == '__main__':
    newsroom_eval()
