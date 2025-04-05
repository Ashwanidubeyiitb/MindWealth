#pragma once
#include "data_types.h"
#include <vector>

class MACDStrategy {
public:
    MACDStrategy(int fastPeriod = 12, int slowPeriod = 26, int signalPeriod = 9);
    
    std::vector<double> calculateMACD(const std::vector<OHLC>& data);
    std::vector<double> calculateSignalLine(const std::vector<double>& macdLine);
    std::vector<double> calculateHistogram(const std::vector<double>& macdLine, const std::vector<double>& signalLine);
    
    StrategyResult runStrategy(const std::vector<OHLC>& data);
    
private:
    int fastPeriod_;
    int slowPeriod_;
    int signalPeriod_;
    
    std::vector<double> calculateEMA(const std::vector<double>& prices, int period);
    double calculateSuccessRate(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals);
    double calculatePerTradeReturn(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals);
};
