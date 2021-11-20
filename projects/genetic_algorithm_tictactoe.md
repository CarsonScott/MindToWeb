# Learning to Play Tic-tac-toe w/ Genetic Algorithms

_Note: this is a project I'm currently working on, mostly to get more comfortable with genetic algorithms but also because it sounded like an interesting (and relatively uncomplicated) problem to sink my teeth into. I hope you feel the same!_

## __The Game__

A game consists of two bots competing with one another to be the first to get 3-in-a-row. Each bot takes a turn placing either an 'x' or an 'o' token into a blank spot on a 3x3 game board until one of them gets 3-in-a-row (horizontal, vertical, or diagonal) and is the winner, or until all the spots are filled on the board in which case the game is a tie.

## __The Bots__

Bots receive an input matrix in which their tokens are represented by 1's and their opponent's by -1's, and empty spots by 0's. Filters are applied over the matrix which look at all of the possible rows (horizontal/vertical/diagonal) and compute an "urgency value". The urgency of a row indicates how close either of the two players are to filling in that row and thus winning the game. The set of all urgency values influences how and what the bot pays attention to when deciding a move.

Once a bot determines the urgency of each row they must decide an action to take (i.e. choose an empty location on the board to place their token). To do this they must take into account the current state of the board along with the urgency values associated with each row. This task is performed by a decision function which outputs a 2D matrix that highlights a single spot on the board.

## __Genetic Minds__

Both the urgency function and decision function are artificial neural networks, the former taking in inputs from part of the game board and producing an urgency value, and the latter taking in inputs from the entire game board as well as the urgency value of each row, and producing an output matrix equal in shape to the game board with a single non-zero value representing the location where the bot wishes to place a token.

These functions (neural networks) are determined genetically and evolve as they are passed down between generations. For any given generation the bots compete in N number of games, the outcomes of which determine the fitness of each individual and therefore determine the following generation. The winning bots are copied and mutated to the replace lower-performing candidates in the next generation.

## __Bot-Bot Evolution__

The process of developing a new population from the existing one involves the use of crossover and mutation. Crossover occurs when the genomes of two high-performing candidates are mixed and combined to form a new, hybrid genome, while mutation refers to the random process of change that happens when genes are carried over into the next generation.

In this case a genome refers to the list of weights/biases of two neural networks (urgency and decision functions). When these genes are carried over there is a small chance that a weight or bias will randomly change in value, which may or may not be advantagous to the recepient (offspring). OVer time these subtle changes cause trial-and-error that pushes the fitness of the population as a whole higher and higher.
