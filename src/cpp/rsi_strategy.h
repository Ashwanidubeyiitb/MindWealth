#pragma once
#include "data_types.h"
#include <vector>

class RSIStrategy {
public:
    RSIStrategy(int period = 14, double overbought = 70.0, double oversold = 30.0);
    
    std::vector<double> calculateRSI(const std::vector<OHLC>& data);
    StrategyResult runStrategy(const std::vector<OHLC>& data);
    
private:
    int period_;
    double overbought_;
    double oversold_;
    
    double calculateSuccessRate(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals);
    double calculatePerTradeReturn(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals);
};
