#pragma once
#include "data_types.h"
#include <vector>

class SupertrendStrategy {
public:
    SupertrendStrategy(int period = 10, double multiplier = 3.0);
    
    std::vector<double> calculateSupertrend(const std::vector<OHLC>& data);
    StrategyResult runStrategy(const std::vector<OHLC>& data);
    
private:
    int period_;
    double multiplier_;
    
    std::vector<double> calculateATR(const std::vector<OHLC>& data);
    double calculateSuccessRate(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals);
    double calculatePerTradeReturn(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals);
};
