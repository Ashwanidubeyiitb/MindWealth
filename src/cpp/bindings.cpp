#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "macd_strategy.h"
#include "rsi_strategy.h"
#include "supertrend_strategy.h"
#include "data_types.h"

namespace py = pybind11;

PYBIND11_MODULE(trading_strategies, m) {
    m.doc() = "Trading strategies implemented in C++";
    
    // Bind OHLC struct
    py::class_<OHLC>(m, "OHLC")
        .def(py::init<>())
        .def_readwrite("open", &OHLC::open)
        .def_readwrite("high", &OHLC::high)
        .def_readwrite("low", &OHLC::low)
        .def_readwrite("close", &OHLC::close)
        .def_readwrite("date", &OHLC::date)
        .def_readwrite("volume", &OHLC::volume);
    
    // Bind TradeSignal struct and Action enum
    py::enum_<TradeSignal::Action>(m, "Action")
        .value("BUY", TradeSignal::Action::BUY)
        .value("SELL", TradeSignal::Action::SELL)
        .value("HOLD", TradeSignal::Action::HOLD)
        .export_values();
    
    py::class_<TradeSignal>(m, "TradeSignal")
        .def(py::init<>())
        .def_readwrite("action", &TradeSignal::action)
        .def_readwrite("date", &TradeSignal::date)
        .def_readwrite("price", &TradeSignal::price);
    
    // Bind StrategyResult struct
    py::class_<StrategyResult>(m, "StrategyResult")
        .def(py::init<>())
        .def_readwrite("signals", &StrategyResult::signals)
        .def_readwrite("successRate", &StrategyResult::successRate)
        .def_readwrite("perTradeReturn", &StrategyResult::perTradeReturn)
        .def_readwrite("numTrades", &StrategyResult::numTrades);
    
    // Bind MACD Strategy
    py::class_<MACDStrategy>(m, "MACDStrategy")
        .def(py::init<int, int, int>(), py::arg("fastPeriod") = 12, py::arg("slowPeriod") = 26, py::arg("signalPeriod") = 9)
        .def("calculateMACD", &MACDStrategy::calculateMACD)
        .def("calculateSignalLine", &MACDStrategy::calculateSignalLine)
        .def("calculateHistogram", &MACDStrategy::calculateHistogram)
        .def("runStrategy", &MACDStrategy::runStrategy);
    
    // Bind RSI Strategy
    py::class_<RSIStrategy>(m, "RSIStrategy")
        .def(py::init<int, double, double>(), py::arg("period") = 14, py::arg("overbought") = 70.0, py::arg("oversold") = 30.0)
        .def("calculateRSI", &RSIStrategy::calculateRSI)
        .def("runStrategy", &RSIStrategy::runStrategy);
    
    // Bind Supertrend Strategy
    py::class_<SupertrendStrategy>(m, "SupertrendStrategy")
        .def(py::init<int, double>(), py::arg("period") = 10, py::arg("multiplier") = 3.0)
        .def("calculateSupertrend", &SupertrendStrategy::calculateSupertrend)
        .def("runStrategy", &SupertrendStrategy::runStrategy);
}
