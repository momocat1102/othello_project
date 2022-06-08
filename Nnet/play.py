from AIGamePlatform import Othello
import time

from utils import *
from MCTS import MCTS
import numpy as np
from othello.OthelloGame import OthelloGame
from othello.keras.NNet import NNetWrapper as NNet

app=Othello() # 會開啟瀏覽器登入Google Account，目前只接受@mail1.ncnu.edu.tw及@mail.ncnu.edu.tw

# def BOT_action(board, color):
#     g = OthelloGame(8)
#     n1 = NNet(g)
#     n1.load_checkpoint('./temp/','best.h5')
#     args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#     mcts1 = MCTS(g, n1, args1)
#     n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
#     position = n1p(g.getCanonicalForm(board, color))
#     position=(position//g.n, position%g.n)
#     return position


@app.competition(competition_id='test3')
def _callback_(board, color): # 函數名稱可以自訂，board是當前盤面，color代表黑子或白子
    # time.sleep(0.5)
    bordsize = 8
    g = OthelloGame(bordsize)
    n1 = NNet(g)
    n1.load_checkpoint('./temp/','temp.h5')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    position = n1p(g.getCanonicalForm(board, color))
    position=(position//bordsize, position%bordsize)
    # print(position)
    return position
     # 回傳要落子的座標

