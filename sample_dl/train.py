from othello import OthelloGame
from othello.bots.DeepLearning import BOT

class Human:
    def getAction(self, game, color):
        print('input coordinate:', end='')
        coor=input()
        return (int(coor[1])-1, ord(coor[0])-ord('A'))
BOARD_SIZE=8


# bot=BOT(board_size=BOARD_SIZE)
# bot2=BOT(board_size=BOARD_SIZE)
# args={
#     'num_of_generate_data_for_train': 100,
#     'epochs': 10,
#     'batch_size': 8,
#     'verbose': True
# }
Iter = 20
compare = 3
bot = BOT(board_size=BOARD_SIZE)
args={
    'num_of_generate_data_for_train': 5,
    'epochs': 2,
    'batch_size': 2,
    'verbose': False
}
bot2 = BOT(board_size=BOARD_SIZE)
# bot.self_play_train(args)

# game=OthelloGame(BOARD_SIZE)
for i in range(Iter):
    print('Iter：' + str(i))
    bot.model.change_name('model_' + str(i) + '_8x8.h5')
    print(bot.model.model_name)
    if i != 0:
        bot.model.load_choise_weights('model_' + str(i-1) + '_8x8.h5')
    bot.self_play_train(args)
    if(i != 0):
        new = 0
        old = 0
        bot.model.load_choise_weights('model_' + str(i) + '_8x8.h5')
        bot2.model.load_choise_weights('model_' + str(i-1) + '_8x8.h5')
        
        for j in range(compare):
            game1=OthelloGame(BOARD_SIZE)
            game2=OthelloGame(BOARD_SIZE)
            if game1.play(black=bot, white=bot2, verbose=False):
                new+=1
            else:
                old+=1
            if game2.play(black=bot2, white=bot, verbose=False):
                old+=1
            else:
                new+=1
        print(new, old)
        print('勝率:')
        print(new/(old + new))
        if new/(old + new) >= 0.6:
            bot.model.save_best_weights()
        else:
            bot2.model.save_best_weights()


# play(self, black, white, verbose=True)
# game=OthelloGame(BOARD_SIZE)
# game.play(black=bot, white=bot2)

