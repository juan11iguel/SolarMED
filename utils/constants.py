#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:05:43 2023

@author: patomareao
"""

colors = ["#E77C8D", "#5AA9A2"]
colors_gnome = {
    "blues": ["#99c1f1", "#62a0ea", "#3584e4", "#1c71d8", "#1a5fb4"],
    "greens": ["#8ff0a4", "#57e389", "#33d17a", "#2ec27e", "#26a269"],
    "yellows": ["#f9f06b", "#f8e45c", "#f6d32d", "#f5c211", "#e5a50a"],
    "oranges": ["#ffbe6f", "#f8e45c", "#ff7800", "#e66100", "#c64600"],
    "reds": ["", "", "", "", ""],
    "purples": ["#dc8add", "#c061cb", "#9141ac", "#813d9c", "#613583"],
    "greys": ["#ffffff", "#f6f5f4", "#deddda", "#c0bfbc", "#9a9996"],
    "blacks": ["", "", "", "", ""],
    "browns": ["", "", "", "", ""]
}


vars_info = {
    # General
    "time": {
        "signal_id": "TimeStamp",
        "label": "Time"
    },
    "Tamb": {
        "signal_id": "TT-DES-030",
        "unit_cs": 'ºC',
        "label": r"$T_{amb}$"
    },
    "I": {
        "signal_id": "RE-SF-001",
        "unit_cs": 'W/m²',
        "label": r"$I$"
    },

    # Thermal storage
    "Tts_h_t": {
        "signal_id": "TT-SF-004",
        "unit_cs": 'ºC',
        "label": r"$T_{ts,h,t}$"
    },
    "Tts_h_m": {
        "signal_id": "TT-SF-005",
        "unit_cs": 'ºC',
        "label": r"$T_{ts,h,m}$"
    },
    "Tts_h_b": {
        "signal_id": "TT-SF-006",
        "unit_cs": 'ºC',
        "label": r"$T_{ts,h,b}$"
    },
    "Tts_c_t": {
        "signal_id": "TT-SF-001",
        "unit_cs": 'ºC',
        "label": r"$T_{ts,c,t}$"
    },
    "Tts_c_m": {
        "signal_id": "TT-SF-002",
        "unit_cs": 'ºC',
        "label": r"$T_{ts,c,m}$"
    },
    "Tts_c_b": {
        "signal_id": "TT-SF-003",
        "unit_cs": 'ºC',
        "label": r"$T_{ts,c,b}$"
    },
    "Tts_t_in": {
        "signal_id": "TT-SF-008",
        "unit_cs": 'ºC',
        "label": r"$T_{ts,t,in}$"
    },
    "Tts_b_in": {
        "signal_id": "TT-AQU-109",
        "unit_cs": 'ºC',
        "label": r"$T_{ts,b,in}$"
    },
    "mts_src": {
        "signal_id": "FT-SF-001",
        "unit_cs": 'm³/h',
        "label": r"$m_{ts,src}$"
    },
    "mts_dis": {
        "signal_id": "FT-AQU-101",
        "unit_cs": 'm³/h',
        "label": r"$m_{ts,dis}$"
    },
    
    # Three-way valve
    "T3wv_src2": {
        "signal_id": "TT-SF-004",
        "unit_cs": 'ºC',
        "label": r"$T_{3wv,src \: (TS)}$",
        "color": "#b85450"
    },
    "T3wv_dis_in": {
        "signal_id": "TT-AQU-107a",
        "unit_cs": 'ºC',
        "label": r"$T_{3wv,dis,in}$",
        "color": "#dd8452"
    },
    "T3wv_dis_out": {
        "signal_id": "HW1TT21",
        "unit_cs": 'ºC',
        "label": r"$T_{3wv,dis,out \: (MED)}$",
        "color": "#7a98c4"
    },
    "m_3wv_dis": {
        "signal_id": "FT-AQU-100",
        "unit_cs": 'm³/h',
        "label": r"$m_{3wv,dis}$"
    },
    "m_3wv_src": {
        "signal_id": "FT-AQU-101",
        "unit_cs": 'm³/h',
        "label": r"$m_{3wv,src}$",
        "color": "#b85450"
    },
    "R_3wv": {
        "signal_id": "ZC-AQU-TCV102",
        "unit_cs": r'%',
        "label": r"$R_{3wv}$"
    },
    "T_3wv_dis_out2": {
        "signal_id": "TT-AQU-109",
        "unit_cs": 'ºC',
        "label": r"$T_{3wv,dis,out}$",
        "color": "#7a98c4"
    },
    "T_3wv_src": {
        "signal_id": "TT-AQU-106",
        "unit_cs": 'ºC',
        "label": r"$T_{3wv,src}$",
        "color": "#b85450"
    },
    
    
    # Solar field
    "Tsf_l2_in": {
        "signal_id": "TT-SF-018",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,l2,in}$"
    },
    "Tsf_l2_out": {
        "signal_id": "TT-SF-019",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,l2,out}$"
    },
    "q_sf_l2": {
        "signal_id": "FT-SF-005",
        "unit_cs": 'm³/h',
        "label": r"$q_{sf,l2}$"
    },
    "Tsf_l3_in": {
        "signal_id": "TT-SF-024",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,l3,in}$"
    },
    "Tsf_l3_out": {
        "signal_id": "TT-SF-025",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,l3,out}$"
    },
    "q_sf_l3": {
        "signal_id": "FT-SF-006",
        "unit_cs": 'm³/h',
        "label": r"$q_{sf,l3}$"
    },
    "Tsf_l4_in": {
        "signal_id": "TT-SF-030",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,l4,in}$"
    },
    "Tsf_l4_out": {
        "signal_id": "TT-SF-031",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,l4,out}$"
    },
    "q_sf_l4": {
        "signal_id": "FT-SF-007",
        "unit_cs": 'm³/h',
        "label": r"$q_{sf,l4}$"
    },
    "Tsf_l5_in": {
        "signal_id": "TT-SF-036",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,l5,in}$"
    },
    "Tsf_l5_out": {
        "signal_id": "TT-SF-037",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,l5,out}$"
    },
    "q_sf_l5": {
        "signal_id": "FT-SF-008",
        "unit_cs": 'm³/h',
        "label": r"$q_{sf,l5}$"
    },
    "q_sf": {
        "signal_id": "FT-SF-002",
        "unit_cs": 'm³/h',
        "label": r"$q_{sf}$"
    },
    "Tsf_in": {
        "signal_id": "TT-SF-009",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,in}$"
    },
    "Tsf_out": {
        "signal_id": "TT-SF-010",
        "unit_cs": 'ºC',
        "label": r"$T_{sf,out}$"
    },
    
    # MED
    "Tmed_s_in": {
        "signal_id": "TT-AQU-algo",
        "unit_cs": 'ºC',
        "label": r"$T_{med,s,in}$"
    },
    "Tmed_c_out": {
        "signal_id": "completar",
        "unit_cs": 'ºC',
        "label": r"$T_{med,c,out}$"
    },
    "mmed_d": {
        "signal_id": "completar?",
        "unit_cs": 'm³/h',
        "label": r"$q_{med,d}$"
    },
    "mmed_s": {
        "signal_id": "completar?",
        "unit_cs": 'm³/h',
        "label": r"$q_{med,s}$"
    },
    "mmed_f": {
        "signal_id": "completar?",
        "unit_cs": 'm³/h',
        "label": r"$q_{med,f}$"
    },
    "mmed_c": {
        "signal_id": "completar?",
        "unit_cs": 'm³/h',
        "label": r"$q_{med,c}$"
    },
    "Tmed_s_out": {
        "signal_id": "TT-AQU-algo",
        "unit_cs": 'ºC',
        "label": r"$T_{med,s,out}$"
    },
    "Tmed_c_in": {
        "signal_id": "?",
        "unit_cs": 'ºC',
        "label": r"$T_{med,c,in}$"
    },
    "wmed_f": {
        "signal_id": "?",
        "unit_cs": 'ºC',
        "label": r"$w_{med,f}$"
    },
    "STEC_med": {
        "signal_id": "?",
        "unit_cs": 'kWhth/m³',
        "label": r"$STEC_{med}$"
    },
    "SEEC_med": {
        "signal_id": "?",
        "unit_cs": 'kWhe/m³',
        "label": r"$SEEC_{med}$"
    },
}




# Habría que invertir keys por values para que no haya nombres duplicados
var_names = {
    # General
    "time": "TimeStamp",
    "Tamb": "TT-DES-030",
    
    # Thermal storage
    "Tts_h_t": "TT-SF-004",
    "Tts_h_m": "TT-SF-005",
    "Tts_h_b": "TT-SF-006",
    "Tts_c_t": "TT-SF-001",
    "Tts_c_m": "TT-SF-002",
    "Tts_c_b": "TT-SF-003",
    "Tts_t_in": "TT-SF-008",
    "Tts_b_in": "TT-AQU-109",
    "m_ts_src": "FT-SF-001",
    "m_ts_dis": "FT-AQU-101",
             
    # Three way valve
    "T_3wv_src":"TT-SF-004", 
    "T_3wv_dis_in":"TT-AQU-107a",
    "T_3wv_dis_out":"HW1TT21", 
    "m_3wv_dis":"FT-AQU-100",
    "m_3wv_src": "FT-AQU-101", 
    "R_3wv": "ZT-AQU-TCV102",
    
    # Heat exchanger
    
    
    # Solar field
}

var_labels = {
    # Thermal storage
    "Tts_h_t": r"$T_{ts,h,t}$",
    "Tts_h_m": r"$T_{ts,h,m}$",
    "Tts_h_b": r"$T_{ts,h,b}$",
    "Tts_c_t": r"$T_{ts,c,t}$",
    "Tts_c_m": r"$T_{ts,c,m}$",
    "Tts_c_b": r"$T_{ts,c,b}$",
    "m_ts_dis": r"$\dot{m}_{dis}$",
    "m_ts_src": r"$\dot{m}_{src}$",
    "Tts_t_in": r"$T_{ts,t,in}$",
    
    # Three way valve
    "T_3wv_src": r"$T_{3wv,src}$",
    "T_3wv_dis_in": r"$T_{3wv,dis,in}$",
    "T_3wv_dis_out": r"$T_{3wv,dis,out}$",
    "m_3wv_dis": r"$\dot{m}_{3wv,dis}$",
    "m_3wv_src": r"$\dot{m}_{3wv,src}$",
    "R_3wv": r"$R_{3wv}$",
    
    # Heat exchanger
    
    
    # Solar field
}