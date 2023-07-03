from data_preprocessing import preprocess_data, generate_indicators

class TradingEnvironment:
    def __init__(self, initial_cash_balance=10000.0, transaction_cost=0.01, data_source='russell_2000_daily.csv'):
        # Initializing the cash balance and portfolio
        self.initial_cash_balance = initial_cash_balance
        self.cash_balance = self.initial_cash_balance
        self.portfolio = {}  # Dictionary to hold current portfolio state
        self.transaction_cost = transaction_cost  # Added transaction cost
        self.data_source = data_source

        # Load market data
        self.market_data = self.load_market_data(self.data_source)

        # Initializing the state
        self.state = self.initialize_state()

        # Define the action space
        self.action_space = self.define_action_space()

    def load_market_data(self, data_source):
        # Load the market data from a file or a database
        # Return it as a pandas DataFrame
        data = preprocess_data(data_source)
        data = generate_indicators(data)
        return data

    def initialize_state(self):
        # Initialize the portfolio and cash balance
        self.portfolio = {}  # Dict with structure: {"Stock Symbol": Number of shares held}
        self.cash_balance = self.initial_cash_balance

        # Initialize the market state
        self.market_state = self.market_data.iloc[0]

        # Initialize performance metrics
        self.performance_metrics = self.calculate_initial_metrics()

        # Initialize technical indicator settings
        self.indicator_settings = self.initialize_indicator_settings()

        # Initialize previous action
        self.previous_action = None
        
          # Initialize prices and counters
        self.buy_prices = {}  # Structure: {'symbol': [price1, price2, ...]}
        self.sell_prices = {}  # Structure: {'symbol': [price1, price2, ...]}
        self.winning_trades = 0
        self.total_trades = 0

        # Concatenate portfolio state, cash balance and market state into full state
        self.state = self.concatenate_state()
        

        return self.state
    
    def concatenate_state(self):
        # Convert portfolio and cash_balance into a compatible format with market_state
        portfolio_vector = self.portfolio_to_vector()
        cash_balance_vector = np.array([self.cash_balance])

        # Convert performance metrics, indicator settings and previous action to a compatible format
        performance_vector = self.metrics_to_vector(self.performance_metrics)
        indicator_vector = self.indicator_settings_to_vector(self.indicator_settings)
        action_vector = self.action_to_vector(self.previous_action)

        # Concatenate all components of the state
        full_state = np.concatenate([portfolio_vector, cash_balance_vector, self.market_state.values, performance_vector, indicator_vector, action_vector])

        return full_state
    
    def portfolio_to_vector(self):
        # Convert portfolio dictionary to a vector (array), compatible with the rest of the state
        # Assume that the stocks are ordered in the same order as in the market_data
        portfolio_vector = []

        for stock_symbol in self.market_data.columns:
            if stock_symbol in self.portfolio:
                portfolio_vector.append(self.portfolio[stock_symbol])
            else:
                portfolio_vector.append(0)

        return np.array(portfolio_vector)
    
    def metrics_to_vector(self, metrics):
        # Convert metrics dictionary to a vector (array), compatible with the rest of the state
        # This function simply extracts the values and forms a numpy array
        metrics_vector = np.array(list(metrics.values()))
        return metrics_vector

    def indicator_settings_to_vector(self, settings):
        # Convert indicator settings to a vector (array)
        # For simplicity, let's say settings are represented by a single scalar value for each indicator
        settings_vector = np.array(list(settings.values()))
        return settings_vector

    def action_to_vector(self, action):
        # Convert previous action to a vector (array)
        # For simplicity, let's say each action is represented by a single scalar value
        # If the action is None (e.g., at the beginning), return a zero vector
        if action is None:
            return np.zeros(len(self.action_space))
        else:
            action_vector = np.array(list(action.values()))
            return action_vector

    def define_action_space(self):
        # Define the action space
        # In DDPG, action_space is usually an interval in real numbers (continuous action space).
        # In this case, you might want to map it to the respective action parameters, like percentage to buy/sell.
        self.action_space = {
            'Buy': [-1.0, 1.0],  # Buy x%, normalized to [-1, 1]
            'Sell': [-1.0, 1.0],  # Sell x%, normalized to [-1, 1]
            'Change Indicator Settings': [-1.0, 1.0]  # Assume this value would be used to adjust indicator settings
        }
        return self.action_space

    def step(self, action):
        # Here, we would update the state of the environment based on the action taken by the agent
        # We would also calculate the reward and determine whether the episode has ended
        # In this example, I am simply moving to the next step in the market data for the next state
        # The specific implementation would depend on the specifics of your problem
        

        self.state = self.market_data.iloc[self.current_step].values
        self.current_step += 1
        
         # Update portfolio and cash balance based on the action
        self.update_portfolio_and_balance(action)

        # Get the new market state
        self.market_state = self.market_data.iloc[self.current_step]

        # Update performance metrics
        self.performance_metrics = self.update_metrics()

        # Update technical indicator settings
        self.indicator_settings = self.update_indicator_settings(action)

        # Update previous action
        self.previous_action = action

        # Update full state
        self.state = self.concatenate_state()

        # For now, I am not updating the portfolio or calculating the reward
        # You would need to fill in these details based on your problem specifics

        reward = self.calculate_reward().

        done = self.current_step >= len(self.market_data)

        return self.state, reward, done

    def reset(self):
        # Reset the environment to the initial state
        self.state = self.market_data.iloc[0].values
        self.current_step = 1
        return self.state

    def calculate_reward(self):
        # Compute the return
        ret = (self.current_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value

        # Compute a measure of risk
        risk = np.std(self.portfolio_value_history)

        # Compute the risk-adjusted return
        risk_adjusted_return = ret / risk

        # Compute a penalty for trading
        trade_penalty = self.transaction_cost * (self.current_portfolio_value != self.previous_portfolio_value)

        # Compute an improvement bonus
        improvement_bonus = 0.0
        if self.performance_metrics['Portfolio Value'] > self.performance_metrics['Running Average Value']:
            improvement_bonus = 0.1  # This value could be adjusted as per requirements

        # The reward is the risk-adjusted return minus the trade penalty plus the improvement bonus
        reward = risk_adjusted_return - trade_penalty + improvement_bonus

        return reward
    
    def update_portfolio_and_balance(self, action):
        # Check the validity of the action
        if not self.is_valid_action(action):
            return

        symbol, action_type, amount = action['symbol'], action['type'], action['amount']
        current_price = self.market_state[symbol]

        if action_type == 'buy':
            cost = current_price * amount

            # If sufficient balance is available, update portfolio and balance
            if self.balance >= cost:
                if symbol not in self.portfolio:
                    self.portfolio[symbol] = 0
                self.portfolio[symbol] += amount
                self.balance -= cost

                # Store buy price
                if symbol not in self.buy_prices:
                    self.buy_prices[symbol] = []
                self.buy_prices[symbol].append(current_price)

        elif action_type == 'sell':
            # If sufficient stocks are available in the portfolio, update portfolio and balance
            if symbol in self.portfolio and self.portfolio[symbol] >= amount:
                self.portfolio[symbol] -= amount
                if self.portfolio[symbol] == 0:
                    del self.portfolio[symbol]
                self.balance += current_price * amount

                # Store sell price and count winning trades
                if symbol not in self.sell_prices:
                    self.sell_prices[symbol] = []
                self.sell_prices[symbol].append(current_price)
                if self.sell_prices[symbol][-1] > self.buy_prices[symbol][0]:  # FIFO strategy
                    self.winning_trades += 1
                del self.buy_prices[symbol][0]  # Remove the corresponding buy price

        # Increment total trades counter
        self.total_trades += 1

        self.update_market_state()  # Update the market state after taking the action

    def calculate_initial_metrics(self):
        # Calculate initial metrics
        self.performance_metrics = {
            'Portfolio Value': self.calculate_portfolio_value(),
            'Running Average Value': self.calculate_portfolio_value(),  # Add this new metric
            'Drawdown': 0,  # No drawdown at the start
            'Winning Trades': 0,  # No trades at the start
            'Total Trades': 0  # No trades at the start
        }

    def initialize_indicator_settings(self):
        # Here we're assuming that the indicator settings are simply the periods of some moving averages
        # They are represented as a dictionary {'SMA': 30, 'EMA': 15}
        self.indicator_settings = {'SMA': 30, 'EMA': 15}

    def update_metrics(self):
        # Update metrics
        current_portfolio_value = self.calculate_portfolio_value()
        running_average_value = (self.performance_metrics['Running Average Value'] * (self.current_step - 1) + current_portfolio_value) / self.current_step
        self.performance_metrics = {
            'Portfolio Value': current_portfolio_value,
            'Running Average Value': running_average_value,
            'Drawdown': self.calculate_drawdown(current_portfolio_value),
            'Winning Trades': self.calculate_winning_trades(),  # This needs more implementation details
            'Total Trades': self.calculate_total_trades()  # This needs more implementation details
        }

    def update_indicator_settings(self, action):
        # Here we're assuming that the 'Change Indicator Settings' action modifies the periods of the moving averages
        # The action is a dictionary with structure {'SMA': 1, 'EMA': -1}, where positive numbers mean increase, and negative numbers mean decrease
        if 'Change Indicator Settings' in action:
            for indicator, change in action['Change Indicator Settings'].items():
                self.indicator_settings[indicator] += change
                
    def calculate_portfolio_value(self):
        return sum(self.market_state[symbol] * amount for symbol, amount in self.portfolio.items())
    
    def calculate_portfolio_value(self):
        return sum(self.market_state[symbol] * amount for symbol, amount in self.portfolio.items())

    def calculate_drawdown(self):
        if not hasattr(self, 'historical_peaks'):
            self.historical_peaks = self.calculate_portfolio_value()
        current_value = self.calculate_portfolio_value()
        self.historical_peaks = max(self.historical_peaks, current_value)
        drawdown = (self.historical_peaks - current_value) / self.historical_peaks
        return drawdown

    def calculate_winning_trades(self):
        return self.winning_trades

    def calculate_total_trades(self):
        return self.total_trades

