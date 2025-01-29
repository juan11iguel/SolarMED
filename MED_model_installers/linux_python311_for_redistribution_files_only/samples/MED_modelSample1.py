#!/usr/bin/env python
"""
Sample script that uses the MED_model module created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

import MED_model
# Import the matlab module only after you have imported
# MATLAB Compiler SDK generated Python modules.
import matlab

my_MED_model = MED_model.initialize()

MsIn = matlab.double([14.0], size=(1, 1))
TsinIn = matlab.double([74.0], size=(1, 1))
MfIn = matlab.double([8.0], size=(1, 1))
TcwoutIn = matlab.double([30.0], size=(1, 1))
TcwinIn = matlab.double([20.0], size=(1, 1))
op_timeIn = matlab.double([0.0], size=(1, 1))
MprodOut, TsoutOut, McwOut, STECOut, SEECOut = my_MED_model.MED_model(MsIn, TsinIn, MfIn, TcwoutIn, TcwinIn, op_timeIn, nargout=5)
print(MprodOut, TsoutOut, McwOut, STECOut, SEECOut, sep='\n')

my_MED_model.terminate()
