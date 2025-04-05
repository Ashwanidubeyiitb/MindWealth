#include "macd_strategy.h"
#include <numeric>
#include <algorithm>

MACDStrategy::MACDStrategy(int fastPeriod, int slowPeriod, int signalPeriod)
    : fastPeriod_(fastPeriod), slowPeriod_(slowPeriod), signalPeriod_(signalPeriod) {}

std::vector<double> MACDStrategy::calculateEMA(const std::vector<double>& prices, int period) {
    std::vector<double> ema(prices.size());
    
    // Simple moving average for the first value
    double sum = 0;
    for (int i = 0; i < period && i < prices.size(); i++) {
        sum += prices[i];
    }
    
    ema[period - 1] = sum / period;
    
    // Calculate EMA for the rest
    double multiplier = 2.0 / (period + 1);
    for (size_t i = period; i < prices.size(); i++) {
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }
    
    return ema;
}

std::vector<double> MACDStrategy::calculateMACD(const std::vector<OHLC>& data) {
    std::vector<double> prices;
    for (const auto& candle : data) {
        prices.push_back(candle.close);
    }
    
    std::vector<double> fastEMA = calculateEMA(prices, fastPeriod_);
    std::vector<double> slowEMA = calculateEMA(prices, slowPeriod_);
    
    std::vector<double> macdLine(prices.size());
    for (size_t i = 0; i < prices.size(); i++) {
        if (i < slowPeriod_ - 1) {
            macdLine[i] = 0;
        } else {
            macdLine[i] = fastEMA[i] - slowEMA[i];
        }
    }
    
    return macdLine;
}

std::vector<double> MACDStrategy::calculateSignalLine(const std::vector<double>& macdLine) {
    return calculateEMA(macdLine, signalPeriod_);
}

std::vector<double> MACDStrategy::calculateHistogram(const std::vector<double>& macdLine, const std::vector<double>& signalLine) {
    std::vector<double> histogram(macdLine.size());
    for (size_t i = 0; i < macdLine.size(); i++) {
        histogram[i] = macdLine[i] - signalLine[i];
    }
    return histogram;
}

StrategyResult MACDStrategy::runStrategy(const std::vector<OHLC>& data) {
    std::vector<TradeSignal> signals;
    
    std::vector<double> macdLine = calculateMACD(data);
    std::vector<double> signalLine = calculateSignalLine(macdLine);
    std::vector<double> histogram = calculateHistogram(macdLine, signalLine);
    
    // Generate buy/sell signals
    bool inPosition = false;
    
    for (size_t i = 1; i < data.size(); i++) {
        // Skip until we have enough data
        if (i <= slowPeriod_ + signalPeriod_) continue;
        
        // Buy signal: MACD line crosses above signal line
        if (!inPosition && histogram[i-1] <= 0 && histogram[i] > 0) {
            TradeSignal signal;
            signal.action = TradeSignal::Action::BUY;
            signal.date = data[i].date;
            signal.price = data[i].close;
            signals.push_back(signal);
            inPosition = true;
        }
        // Sell signal: MACD line crosses below signal line
        else if (inPosition && histogram[i-1] >= 0 && histogram[i] < 0) {
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

double MACDStrategy::calculateSuccessRate(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals) {
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

double MACDStrategy::calculatePerTradeReturn(const std::vector<OHLC>& data, const std::vector<TradeSignal>& signals) {
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
