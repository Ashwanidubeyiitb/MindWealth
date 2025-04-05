#include "rsi_strategy.h"
#include <numeric>
#include <algorithm>
#include <cmath>

RSIStrategy::RSIStrategy(int period, double overbought, double oversold)
    : period_(period), overbought_(overbought), oversold_(oversold) {}

std::vector<double> RSIStrategy::calculateRSI(const std::vector<OHLC>& data) {
    std::vector<double> rsi(data.size(), 0);
    
    if (data.size() <= period_ + 1) {
        return rsi;
    }
    
    std::vector<double> gains(data.size(), 0);
    std::vector<double> losses(data.size(), 0);
    
    // Calculate price changes
    for (size_t i = 1; i < data.size(); i++) {
        double change = data[i].close - data[i-1].close;
        if (change > 0) {
            gains[i] = change;
        } else {
            losses[i] = std::abs(change);
        }
    }
    
    // Calculate first average gain and loss
    double avgGain = std::accumulate(gains.begin() + 1, gains.begin() + period_ + 1, 0.0) / period_;
    double avgLoss = std::accumulate(losses.begin() + 1, losses.begin() + period_ + 1, 0.0) / period_;
    
    // Calculate RSI
    for (size_t i = period_ + 1; i < data.size(); i++) {
        // Smooth averages
        avgGain = (avgGain * (period_ - 1) + gains[i]) / period_;
        avgLoss = (avgLoss * (period_ - 1) + losses[i]) / period_;
        
        if (avgLoss == 0) {
            rsi[i] = 100;
        } else {
            double rs = avgGain / avgLoss;
            rsi[i] = 100 - (100 / (1 + rs));
        }
    }
    
    return rsi;
}

StrategyResult RSIStrategy::runStrategy(const std::vector<OHLC>& data) {
    std::vector<TradeSignal> signals;
    
    std::vector<double> rsi = calculateRSI(data);
    
    // Generate buy/sell signals
    bool inPosition = false;
    
    for (size_t i = period_ + 1; i < data.size(); i++) {
        // Buy signal: RSI crosses above oversold level
        if (!inPosition && rsi[i-1] <= oversold_ && rsi[i] > oversold_) {
            TradeSignal signal;
            signal.action = TradeSignal::Action::BUY;
            signal.date = data[i].date;
            signal.price = data[i].close;
            signals.push_back(signal);
            inPosition = true;
        }
        // Sell signal: RSI crosses below overbought level
        else if (inPosition && rsi[i-1] >= overbought_ && rsi[i] < overbought_) {
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

double RSIStrategy::calculateSuccessRate(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals) {
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

double RSIStrategy::calculatePerTradeReturn(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals) {
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
