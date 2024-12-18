from utils import load_config
from option_pricing_PINN import OptionPricingSolver

def main():
    # Load configuration from YAML file
    # config = load_config('./Configs/Heston_config.yaml')
    
    config = load_config('./Configs/BS_config.yaml')
    solver = OptionPricingSolver(config)
    solver.train()
    solver.save_model()
    solver.evaluate()
    solver.greeks.plot_greeks()
    solver.greeks.plot_implied_volatility()

if __name__ == "__main__":
    main()