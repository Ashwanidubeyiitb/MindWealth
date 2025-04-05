#pragma once
#include <vector>
#include <string>
#include <utility>

struct OHLC {
    double open;
    double high;
    double low;
    double close;
    std::string date;
    double volume;
};

struct TradeSignal {
    enum class Action {
        BUY,
        SELL,
        HOLD
    };
    
    Action action;
    std::string date;
    double price;
};

struct StrategyResult {
    std::vector<TradeSignal> signals;
    double successRate;
    double perTradeReturn;
    int numTrades;
};
