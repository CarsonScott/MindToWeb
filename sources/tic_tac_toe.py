import numpy as np 

ROW_INDICES = [
	[(0,0), (0,1), (0,2)],
	[(1,0), (1,1), (1,2)],
	[(2,0), (2,1), (2,2)],

	[(0,0), (1,0), (2,0)],
	[(0,1), (1,1), (2,1)],
	[(0,2), (1,2), (2,2)],

	[(0,0), (1,1), (2,2)],
	[(0,2), (1,1), (2,0)]]

def all_equal(X):
    for i in range(1, len(X)):
        if X[i] != X[i-1]:
            return False
    return True

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def clamp(x, low, high):
	return low if x < low else high if x > high else x

def crossover(net1, net2):
	weights1 = net1.weights
	weights2 = net2.weights

	biases1 = net1.biases
	biases2 = net2.biases

	weights = weights1.copy()
	biases = biases1.copy()

	for i in range(len(biases1)):
		for j in range(len(biases1[i])):
			if np.random.randint(2) == 0:
				bias = biases1[i][j]
			else: bias = biases2[i][j]
			biases[i][j] = bias

	for i in range(len(weights1)):
		for j in range(len(weights1[i])):
			for k in range(len(weights1[i][j])):
				if np.random.randint(2) == 0:
					weight = weights1[i][j][k]
				else: weight = weights2[i][j][k]
				weights[i][j][k] = weight

	layer_sizes = net1.layer_sizes

	network = Network(layer_sizes, weights=weights, biases=biases)
	return network

def mutate(network, mutation_probability, mutation_magnitude):
	weights = np.array(network.weights)
	biases = np.array(network.biases)

	for i in range(len(weights)):
		for j in range(len(weights[i])):
			for k in range(len(weights[i][j])):
				if np.random.uniform() < mutation_probability:
					delta = np.random.uniform(-mutation_magnitude, mutation_magnitude)
					weights[i][j][k] = clamp(weights[i][j][k] + delta, *network.weight_range)

	for i in range(len(biases)):
		for j in range(len(biases[i])):
			if np.random.uniform() < mutation_probability:
				delta = np.random.uniform(-mutation_magnitude, mutation_magnitude)
				biases[i][j] = clamp(biases[i][j] + delta, *network.weight_range)

	layer_sizes = network.layer_sizes
	return Network(layer_sizes, weights=weights, biases=biases)
 
def print_board(board):
	string = ''
	for i in range(len(board)):
		for j in range(len(board[i])):
			string += board[i][j] + ' '
		string += '\n'
	print(string)

def get_rows(board):
	rows = np.zeros((len(ROW_INDICES), 3))
	for i in range(len(ROW_INDICES)):
		for j in range(len(ROW_INDICES[i])):
			n,m = ROW_INDICES[i][j]
			rows[i][j] = board[n][m]
	return rows

def is_full(board):
	return '.' not in np.concatenate(board)

def has_winner(board):

	rows = np.full((len(ROW_INDICES), 3), '.')
	
	for i in range(len(ROW_INDICES)):
		for j in range(len(ROW_INDICES[i])):
			n,m = ROW_INDICES[i][j]
			rows[i][j] = board[n][m]

	for row in rows:
		if '.' not in row:
			if all_equal(row):
				return True
	return False



class Network:
	def __init__(self, layer_sizes, weights=None, biases=None):
		input_size = layer_sizes[0]
		output_size = layer_sizes[len(layer_sizes)-1]

		self.layer_sizes = layer_sizes
		self.input_size = input_size
		self.output_size = output_size

		self.outputs = np.array([np.zeros((size,)) for size in layer_sizes])
		self.biases = np.array([np.zeros((size,)) for size in layer_sizes])
		self.weights = np.array([np.zeros((layer_sizes[i], layer_sizes[i-1])) for i in range(1, len(layer_sizes))])

		self.weight_range = (0, 1)

		self.randomize()

		if weights is not None: self.weights = np.array(weights)
		if biases is not None: self.biases = np.array(biases)

	def randomize(self):
		for i in range(len(self.biases)):
			for j in range(len(self.biases[i])):
				self.biases[i][j] = np.random.uniform(*self.weight_range)

		for i in range(len(self.weights)):
			for j in range(len(self.weights[i])):
				for k in range(len(self.weights[i][j])):
					self.weights[i][j][k] = np.random.uniform(*self.weight_range)

	def compute(self, X):
		self.outputs[0] = np.array(X)

		for i in range(1, len(self.outputs)):
			# f = sigmoid if i < len(self.outputs)-1 else np.tanh
			f = sigmoid
			# f = relu if i < len(self.outputs)-1 else np.tanh

			for j in range(len(self.outputs[i])):
				x = sum(self.outputs[i-1] * self.weights[i-1][j])
				b = self.biases[i][j]
				y = f(x + b)
				self.outputs[i][j] = y

		return self.outputs[len(self.outputs)-1]

	def genome(self):
		return {"weights": to_list(self.weights),
				"biases": to_list(self.biases)}



class Bot:
	def __init__(self):
		self.rows = np.zeros((len(ROW_INDICES), 3))

		self.urgency_network = Network([3, 1])
		self.decision_network = Network([3, 3])

		self.urgency_vector = np.zeros((len(ROW_INDICES),))
		self.decision_vector = np.zeros((3,))

		self.selected_row = None
		self.selected_indices = None

	def select(self):
		row = self.rows[self.selected_row]
		output = self.decision_network.compute(row)
		ordered = np.flip(np.argsort(output))

		winner = None 
		for i in ordered:
			if row[i] == 0:
				winner = i

		indices = None
		if winner is not None:
			indices = ROW_INDICES[self.selected_row][winner]

		return indices

	def compute(self, board):
		self.rows = get_rows(board)

		for i in range(len(self.rows)):
			self.urgency_vector[i] = self.urgency_network.compute(self.rows[i])
		
		min_urgency = max(self.urgency_vector) * 0.5

		options = []
		for i in range(len(self.rows)):
			if 0 in self.rows[i] and self.urgency_vector[i] >= min_urgency:
				options.append(i)

		if len(options) == 0:
			options = [i for i in range(len(self.rows))]

		self.selected_row = np.random.choice(options)
		
		return self.select()



class Game:
	def __init__(self, bots):
		self.bots = bots
		self.tokens = ['x', 'o']
		self.index = 0
		self.done = False
		self.board = np.full((3,3), '.')
		self.winner = None

	def reset(self, bots):
		self.bots = bots
		self.index = 0
		self.done = False
		self.board = np.full((3,3), '.')
		self.winner = None

	def convert(self):
		board = np.zeros((3,3))
		token = self.tokens[self.index]
		for i in range(len(self.board)):
			for j in range(len(self.board[i])):
				if self.board[i][j] != '.':
					if self.board[i][j] == token:
						board[i][j] = 1
					else: board[i][j] = -1
		return board

	def update(self):
		if not self.done:
			bot = self.bots[self.index]
			token = self.tokens[self.index]
			board = self.convert()
			output = bot.compute(board)

			if output is not None:
				n,m = output

				self.board[n][m] = token
				
				if has_winner(self.board):
					# print("Winner: " + token)
					self.winner = self.index
					self.done = True 

				elif is_full(self.board):
					# print("Draw")
					self.done = True
				else:
					self.index = int(not self.index)
			else:
				# print("Winner by default: " + self.tokens[int(not self.index)])
				self.winner = int(not self.index)
				self.done = True



class Population:
	def __init__(self, size, games_per_gen, winners_per_gen, losers_per_gen, mutation_rate, mutation_magnitude):
		self.size = size
		self.games_per_gen = games_per_gen
		self.winners_per_gen = winners_per_gen
		self.losers_per_gen = losers_per_gen
		self.mutation_rate = mutation_rate
		self.mutation_magnitude = mutation_magnitude

		self.bots = []
		self.scores = []

		self.games = []
		self.indices = []

		self.games_won = []
		self.games_played = []

		for i in range(self.size):
			self.add_bot(Bot())

	def add_bot(self, bot):
		self.bots.append(bot)
		self.scores.append(0)
		self.games_won.append([])
		self.games_played.append([])

	def update_population(self, winners, losers):
		for loser in losers:
			p1, p2 = [self.bots[i] for i in np.random.choice(winners, 2)]

			urgency_network = crossover(p1.urgency_network, p2.urgency_network)
			decision_network = crossover(p1.decision_network, p2.decision_network)

			child = Bot()

			child.urgency_network = mutate(urgency_network, self.mutation_rate, self.mutation_magnitude)
			child.decision_network = mutate(decision_network, self.mutation_rate, self.mutation_magnitude)

			self.bots[loser] = child

		for i in range(self.size):
			if i not in winners and i not in losers:
				bot = self.bots[i]
				bot.urgency_network = mutate(bot.urgency_network, self.mutation_rate, self.mutation_magnitude)
				bot.decision_network = mutate(bot.decision_network, self.mutation_rate, self.mutation_magnitude)

	def update(self):
		self.games = []
		self.indices = []

		self.games_played = np.zeros((self.size,))
		self.games_won = np.zeros((self.size,))
		self.scores = np.zeros((self.size,))

		for i in range(self.games_per_gen):
			indices = np.random.choice(list(range(self.size)), 2)
			bots = [self.bots[j] for j in indices]
			game = Game(bots)

			self.indices.append(indices)
			self.games.append(game)

			self.games_played[indices[0]] += 1
			self.games_played[indices[1]] += 1

		done = False
		while not done:
			done = True
			for i in range(len(self.games)):
				game = self.games[i]
				indices = self.indices[i]

				if not game.done:
					game.update()

					if not game.done:
						done = False

					elif game.winner != None:
						winner = indices[game.winner] 
						self.games_won[winner] += 1

		for i in range(self.size):
			self.scores[i] = self.games_won[i] / self.games_played[i]

		ordered = np.flip(np.argsort(self.scores))

		winners = ordered[:self.winners_per_gen]
		losers = ordered[len(ordered)-self.losers_per_gen-1:]

		self.update_population(winners, losers)



if __name__ == "__main__":
	pop = Population(20, 300, 5, 5, 0.01, 0.2)

	# evolve the population
	
	for i in range(100):
		print('GEN: '+str(i))
		pop.update()

	# display the results of test games

	for i in range(5):
		game = Game([pop.bots[i] for i in np.random.choice(list(range(pop.size)), 2)])

		while not game.done:
			game.update()
			print_board(game.board)

		print()
