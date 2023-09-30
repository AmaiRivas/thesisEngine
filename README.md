# thesisEngine

thesisEngine is a chess engine designed to run on the [lichess-bot](https://github.com/ShailChoksi/lichess-bot) framework. To use the enigne, you can replace the lichess-bot strategies.py with this repository's "strategies.py" file, and follow the lichess-bot instructions to connect your lichess bot account.

## Repository Structure

- `README.md`: This file.
- `strategies.py`: The main Python file containing the chess engine logic. Replace this in your lichess-bot directory to run the engine.
- `all_games.pgn`: A PGN file containing all the rated games played by this engine, which were part of the testing.
- `phases_and_games.txt`: A text file indicating the phases of development and until which game each phase was active.

## How to Use

1. Clone the [lichess-bot](https://github.com/ShailChoksi/lichess-bot) repository.
2. Replace the `strategies.py` in the lichess-bot directory with the one from this repository.
3. Follow the instructions in the lichess-bot repository to set up and run the bot.

## Acknowledgments
The code is based on and inspired by the following resources:
- [Bruce Moreland's Programming Pages](https://web.archive.org/web/20071026090003/http://www.brucemo.com/compchess/programming/index.htm)
- [Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)
- [Chess Programming YouTube Channel](https://www.youtube.com/@chessprogramming591)
- [Sunfish](https://github.com/thomasahle/sunfish)
- [Theodora](https://github.com/yigitkucuk/Theodora)
- [pychessengine](https://github.com/perintyler/pychessengine/tree/master)
- [angelfish](https://github.com/VCHui/angelfish)

## APIs, GitHub and Libraries used
[lichess-bot](https://github.com/ShailChoksi/lichess-bot)
[lichess-API](https://lichess.org/api) Used by lichess-bot
[python-chess library](https://python-chess.readthedocs.io/en/latest/)


