#include "supertrend_strategy.h"
#include <numeric>
#include <algorithm>
#include <cmath>

SupertrendStrategy::SupertrendStrategy(int period, double multiplier)
    : period_(period), multiplier_(multiplier) {}

std::vector<double> SupertrendStrategy::calculateATR(const std::vector<OHLC>& data) {
    std::vector<double> trueRanges(data.size(), 0);
    std::vector<double> atr(data.size(), 0);
    
    // Calculate True Range
    for (size_t i = 1; i < data.size(); i++) {
        double highLow = data[i].high - data[i].low;
        double highClose = std::abs(data[i].high - data[i-1].close);
        double lowClose = std::abs(data[i].low - data[i-1].close);
        
        trueRanges[i] = std::max({highLow, highClose, lowClose});
    }
    
    // Calculate first ATR as simple average
    if (data.size() > period_) {
        double sum = std::accumulate(trueRanges.begin() + 1, trueRanges.begin() + period_ + 1, 0.0);
        atr[period_] = sum / period_;
        
        // Calculate smoothed ATR
        for (size_t i = period_ + 1; i < data.size(); i++) {
            atr[i] = (atr[i-1] * (period_ - 1) + trueRanges[i]) / period_;
        }
    }
    
    return atr;
}

std::vector<double> SupertrendStrategy::calculateSupertrend(const std::vector<OHLC>& data) {
    std::vector<double> supertrend(data.size(), 0);
    std::vector<double> atr = calculateATR(data);
    
    std::vector<double> upperBand(data.size(), 0);
    std::vector<double> lowerBand(data.size(), 0);
    std::vector<bool> trend(data.size(), true); // true for uptrend
    
    // Calculate bands
    for (size_t i = period_; i < data.size(); i++) {
        double hl2 = (data[i].high + data[i].low) / 2;
        upperBand[i] = hl2 + multiplier_ * atr[i];
        lowerBand[i] = hl2 - multiplier_ * atr[i];
    }
    
    // Calculate Supertrend
    for (size_t i = period_ + 1; i < data.size(); i++) {
        if (data[i].close > upperBand[i-1]) {
            trend[i] = true;
        } else if (data[i].close < lowerBand[i-1]) {
            trend[i] = false;
        } else {
            trend[i] = trend[i-1];
            
            if (trend[i] && lowerBand[i] < lowerBand[i-1]) {
                lowerBand[i] = lowerBand[i-1];
            }
            
            if (!trend[i] && upperBand[i] > upperBand[i-1]) {
                upperBand[i] = upperBand[i-1];
            }
        }
        
        supertrend[i] = trend[i] ? lowerBand[i] : upperBand[i];
    }
    
    return supertrend;
}

StrategyResult SupertrendStrategy::runStrategy(const std::vector<OHLC>& data) {
    std::vector<TradeSignal> signals;
    
    std::vector<double> supertrend = calculateSupertrend(data);
    std::vector<bool> trend(data.size(), true);
    
    // Determine trend
    for (size_t i = period_ + 1; i < data.size(); i++) {
        trend[i] = data[i].close > supertrend[i];
    }
    
    // Generate buy/sell signals
    bool inPosition = false;
    
    for (size_t i = period_ + 2; i < data.size(); i++) {
        // Buy signal: Trend changes from down to up
        if (!inPosition && !trend[i-1] && trend[i]) {
            TradeSignal signal;
            signal.action = TradeSignal::Action::BUY;
            signal.date = data[i].date;
            signal.price = data[i].close;
            signals.push_back(signal);
            inPosition = true;
        }
        // Sell signal: Trend changes from up to down
        else if (inPosition && trend[i-1] && !trend[i]) {
            TradeSignal signal;
            signal.action = TradeSignal::Action::SELL;
            signal.date = data[i].date;
            signal.price = data[i].close;
            signals.push_back(signal);
            inPosition = false;
        }
    }
    
    // Calculate performance metrics
    double successRate = calculateSuccessRate(data, signals);
    double perTradeReturn = calculatePerTradeReturn(data, signals);
    int numTrades = signals.size() / 2; // Each trade consists of a buy and sell
    
    return {signals, successRate, perTradeReturn, numTrades};
}

double SupertrendStrategy::calculateSuccessRate(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals) {
    int profitableTrades = 0;
    int totalTrades = 0;
    
    for (size_t i = 0; i < signals.size() - 1; i += 2) {
        if (i + 1 >= signals.size()) break;
        
        if (signals[i].action == TradeSignal::Action::BUY && 
            signals[i+1].action == TradeSignal::Action::SELL) {
            totalTrades++;
            if (signals[i+1].price > signals[i].price) {
                profitableTrades++;
            }
        }
    }
    
    return totalTrades > 0 ? (double)profitableTrades / totalTrades * 100.0 : 0.0;
}

double SupertrendStrategy::calculatePerTradeReturn(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals) {
    double totalReturn = 0.0;
    int totalTrades = 0;
    
    for (size_t i = 0; i < signals.size() - 1; i += 2) {
        if (i + 1 >= signals.size()) break;
        
        if (signals[i].action == TradeSignal::Action::BUY && 
            signals[i+1].action == TradeSignal::Action::SELL) {
            totalTrades++;
            double tradeReturn = (signals[i+1].price - signals[i].price) / signals[i].price * 100.0;
            totalReturn += tradeReturn;
        }
    }
    
    return totalTrades > 0 ? totalReturn / totalTrades : 0.0;
}
