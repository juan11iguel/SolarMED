classdef ptOp
  % Class that stores all necesary data for the definition of an operating
  % point of a MED plant gathered from real operation data. It includes
  % methods to export the MATLAB workspace variable into a MongoDB database
  
    properties
        %% Variable parameters

        var_ids            = ["Ms"	"Mf"	"Mprod"	"Mprod_4"	"Mcw"	"Msw"	"DEAHP"	"Ts_in"	"Ts_out"	"Tf"	"Tprod"	"Tbrine"	"Tvc"	"Tcwin"	"Tcwout"	"Tv_ref_1"	"Tv_ref_2"	"Tv_ref_3"	"Tv_ref_4"	"Tv_ref_5"	"Tv_ref_6"	"Tv_ref_7"	"Tv_ref_8"	"Tv_ref_9"	"Tv_ref_10"	"Tv_ref_11"	"Tv_ref_12"	"Tv_ref_13"	"Tv_ref_14"	"Tph_ref_2"	"Tph_ref_3"	"Tph_ref_4"	"Tph_ref_5"	"Tph_ref_6"	"Tph_ref_7"	"Tph_ref_8"	"Tph_ref_9"	"Tph_ref_10"	"Tph_ref_11"	"Tph_ref_12"	"Tph_ref_13"	"Tph_ref_14"	"Tf_pre"	"Tamb"	"Ps" "HRamb"	"Lprod"	"Lbrine"	"Xprod"	"Xf"	"P1"	"P2"	"P3"];
        sensor_ids_antiguo = ["HW1FT20"	"FA015"	"FA016"	"-"	"-"	"FA014"	"-"	"HW1TT20"	"HW1TT21"	"TA003"	"TA007"	"-"	"PA012"	"TA006"	"SW2TC1"	"PA011"	"PA021"	"-"	"PA022"	"-"	"PA023"	"-"	"PA024"	"-"	"PA025"	"-"	"PA026"	"-"	"PA027"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"TA005"	"-"	"-"	"PT101"	"-" "-"	"-"	"-"	"-"	"-"	"-"	"-"];
        sensor_ids_nuevo   = ["FT-AQU-100"	"FT-DES-003"	"FT-DES-005"	"-"	"-"	"FT-DES-002"	"FT-AQU-102"	"TT-AQU-107a"	"HW1TT21"	"TE-DES-001"	"TE-DES-016"	"-"	"PT-DES-015"	"TE-DES-015"	"SW2TC1"	"PT-DES-001"	"PT-DES-002"	"-"	"PT-DES-004"	"-"	"PT-DES-006"	"-"	"PT-DES-008"	"-"	"PT-DES-010"	"-"	"PT-DES-012"	"-"	"PT-DES-014"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"-"	"TE-DES-013"	"-"	"TT-DES-030" "PT-AQU-101"	"HT-DES-030"	"-"	"-"	"-"	"-"	"-"	"-"	"-"];
        sensor_ids_2022    = ["FT-AQU-100"	"FT-DES-003"	"FT-DES-005"	"FT-DES-006"	"FT-DES-002"	"FT-DES-002"	"FT-AQU-102"	"TT-AQU-107a"	"HW1TT21"	"TE-DES-001"	"TE-DES-016"	"TE-DES-017"	"PT-DES-015"	"TE-DES-015"	"SW2TC1"	"PT-DES-001"	"PT-DES-002"	"PT-DES-003"	"PT-DES-004"	"PT-DES-005"	"PT-DES-006"	"PT-DES-007"	"PT-DES-008"	"PT-DES-009"	"PT-DES-010"	"PT-DES-011"	"PT-DES-012"	"PT-DES-013"	"PT-DES-014"	"TT-DES-002"	"TT-DES-003"	"TT-DES-004"	"TT-DES-005"	"TT-DES-006"	"TT-DES-007"	"TT-DES-008"	"TT-DES-009"	"TT-DES-010"	"TT-DES-011"	"TT-DES-012"	"TT-DES-013"	"TT-DES-014"	"TE-DES-020"	"TT-DES-030" "PT-AQU-101" "HT-DES-030"	"LT-DES-001"	"LT-DES-002"	"CT-DES-001"	"-"	"PK-MED-E01-pa"	"PK-MED-E02-pa"	"PK-MED-E03-pa"];
        units_SCADA        = ["L/s"	"m³/h"	"m³/h"	"?"	"m³/h"	"m³/h"	"?"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"mbar"	"ºC"	"ºC"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"mbar"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC" "bar" "%"	"mm"	"mm"	"mSiem"	"-"	"W"	"W"	"W"];
        units_model        = ["kg/s"	"kg/s"	"kg/s"	"kg/s"	"kg/s"	"kg/s"	"?"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC"	"ºC" "bar"	"%"	"mm"	"mm"	"g/kg"	"g/kg"	"KW"	"KW"	"KW"];
        descriptions       = [  "Caudal de agua caliente"
                                "Caudal de agua de alimentación"
                                "Caudal de destilado a la salida del condensador"
                                "Caudal de destilado a la salida del efecto 4"
                                "Caudal de agua de mar bombeada desde la piscina menos el usado por el sistema de vacío (4 m³/h)"
                                "Caudal de agua de mar bombeada desde la piscina"
                                "Caudal que indica que se está operando con la bomba de calor"
                                "Temperatura de entrada de agua caliente"
                                "Temperatura de salida de agua caliente"
                                "Temperatura de salida del primer precalentador"
                                "Temperatura del destilado a la salida del condensador"
                                "Temperatura de salmuera a la salida del último efecto"
                                "Presión de vapor en el condensador"
                                "Temperatura de entrada del agua de mar en el condensador"
                                "Temperatura de salida del agua de mar en el condensador"
                                "Presión en el efecto 1"
                                "Presión en el efecto 2"
                                "Presión en el efecto 3"
                                "Presión en el efecto 4"
                                "Presión en el efecto 5"
                                "Presión en el efecto 6"
                                "Presión en el efecto 7"
                                "Presión en el efecto 8"
                                "Presión en el efecto 9"
                                "Presión en el efecto 10"
                                "Presión en el efecto 11"
                                "Presión en el efecto 12"
                                "Presión en el efecto 13"
                                "Presión en el efecto 14"
                                "Temperatura de salida del precalentador 2"
                                "Temperatura de salida del precalentador 3"
                                "Temperatura de salida del precalentador 4"
                                "Temperatura de salida del precalentador 5"
                                "Temperatura de salida del precalentador 6"
                                "Temperatura de salida del precalentador 7"
                                "Temperatura de salida del precalentador 8"
                                "Temperatura de salida del precalentador 9"
                                "Temperatura de salida del precalentador 10"
                                "Temperatura de salida del precalentador 11"
                                "Temperatura de salida del precalentador 12"
                                "Temperatura de salida del precalentador 13"
                                "Temperatura de entrada al precalentador 13"
                                "Temperatura de salida del agua de alimentación después de pasar por precalentadores con producto/salmuera"
                                "Temperatura ambiente"
                                "Presión a la salida de agua caliente"
                                "Humedad relativa ambiente"
                                "Nivel de destilado en el condensador"
                                "Nivel de salmuera en el último efecto"
                                "Salinidad del destilado a la salida del condensador"
                                "Salinidad del agua de alimentación"
                                "Potencia consumida por bombas de agua caliente (Ms) y agua de alimentación (Mf)"
                                "Potencia consumida por el sistema de vacío y bombas de destilado (Mprod) y salmuera"
                                "Potencia consumida por la bomba de agua mar (Msw)"
                            ];
        % sensor_ids
%         sensor_ids_antiguo  = ["HW1TT20";"HW1TT21";"TA003";"FA015";"FA014";"HW1FT20";"FA016";"TA007";"PA012";"TA006";"SW2TC1";"PT100";"TA005";"PA011";"PA021";"PA022";"PA023";"PA024";"PA025";"PA026";"PA027"];
%         sensor_ids_nuevo    = ["TT-AQU-107a";"HW1TT21";"TE-DES-001";"FT-DES-003";"FT-DES-002";"FT-AQU-100";"FT-DES-005";"TE-DES-016";"PT-DES-015";"TE-DES-015";"SW2TC1";"PT-AQU-101";"TE-DES-013";"PT-DES-001";"PT-DES-002";"PT-DES-004";"PT-DES-006";"PT-DES-008";"PT-DES-010";"PT-DES-012";"PT-DES-014"];
%         var_ids             = ["Ts_in";"Ts_out";"Tf";"Mf";"Msw";"Ms";"Mprod";"Tprod";"Tvc";"Tcwin";"Tcwout";"Ps";"Tph_ref_13";"Tv_ref_1";"Tv_ref_2";"Tv_ref_4";"Tv_ref_6";"Tv_ref_8";"Tv_ref_10";"Tv_ref_12";"Tv_ref_14"];

%         sensor_ids_2022     = ['FT-AQU-100'	'FT-DES-003'	'FT-DES-005'	'FT-DES-006'	'FT-DES-002'	""	'FT-AQU-102'	'TT-AQU-107a'	'HW1TT21'	'TE-DES-001'	'TE-DES-016'	'TE-DES-017'	'PT-DES-015'	'TE-DES-015'	'SW2TC1'	'PT-DES-001'	'PT-DES-002'	'PT-DES-003'	'PT-DES-004'	'PT-DES-005'	'PT-DES-006'	'PT-DES-007'	'PT-DES-008'	'PT-DES-009'	'PT-DES-010'	'PT-DES-011'	'PT-DES-012'	'PT-DES-013'	'PT-DES-014'	'TT-DES-002'	'TT-DES-003'	'TT-DES-004'	'TT-DES-005'	'TT-DES-006'	'TT-DES-007'	'TT-DES-008'	'TT-DES-009'	'TT-DES-010'	'TT-DES-011'	'TT-DES-012'	'TT-DES-013'	'TT-DES-014'	'TE-DES-020'	'TT-DES-030'	'HT-DES-030'	'LT-DES-001'	'LT-DES-002'	'CT-DES-001'	""	'PK-MED-E01-ea'	'PK-MED-E02-ea'	'PK-MED-E03-ea'];
%         new_var_ids_2022    = ["Ms"	"Mf"	"Mprod"	"Mprod_4"	"Mcw"	"Msw"	"DEAHP"	"Ts_in"	"Ts_out"	"Tf"	"Tprod"	"Tbrine"	"Tvc"	"Tcwin"	"Tcwout"	"Tv_ref_1"	"Tv_ref_2"	"Tv_ref_3"	"Tv_ref_4"	"Tv_ref_5"	"Tv_ref_6"	"Tv_ref_7"	"Tv_ref_8"	"Tv_ref_9"	"Tv_ref_10"	"Tv_ref_11"	"Tv_ref_12"	"Tv_ref_13"	"Tv_ref_14"	"Tph_ref_2"	"Tph_ref_3"	"Tph_ref_4"	"Tph_ref_5"	"Tph_ref_6"	"Tph_ref_7"	"Tph_ref_8"	"Tph_ref_9"	"Tph_ref_10"	"Tph_ref_11"	"Tph_ref_12"	"Tph_ref_13"	"Tph_ref_14"	"Tf_pre"	"Tamb"	"HRamb"	"Lprod"	"Lbrine"	"Xprod"	"Xf"	"P1"	"P2"	"P3"];
        % Unit for each of the variables
        % Model units
%         units_model = ["ºC";"ºC";"ºC";"kg/s";"kg/s";"kg/s";"kg/s";"ºC";"ºC";"ºC";"ºC";"Ps";"ºC";"ºC";"ºC";"ºC";"ºC";"ºC";"ºC";"ºC";"ºC"];
        % SCADA units
%         units_SCADA = ["ºC";"ºC";"ºC";"m3/h";"m3/h";"L/s";"m3/h";"ºC";"mbar";"ºC";"ºC";"bar";"ºC";"mbar";"mbar";"mbar";"mbar";"mbar";"mbar";"mbar";"mbar"];
%         units_SCADA = 
        % Variable descriptions
%         descriptions = [
%             "Inlet temperature of the source at the first effect",...
%             "Outlet temperature of the source at the first effect",...
%             "Feedwater temperature at the first effect",...
%             "Feedwater flow at the first effect",...
%             "Seawater flow through the condenser",...
%             "Heat source fluid flow",...
%             "Distillate flow leaving the last effect", ...
%             "Distillate temperature leaving the last effect",...
%             "Steam temperature at the condenser",...
%             "Seawater temperature at the inlet of the condender",...
%             "Seawater temperature at the outlet of the condenser",...
%             "Heat source fluid pressure",...
%             "Feedwater temperature at the inlet of the last preheater",...
%             "Vapour temperature at the 1st effect",...
%             "Vapour temperature at the 2nd effect",...
%             "Vapour temperature at the 4th effect",...
%             "Vapour temperature at the 6th effect",...
%             "Vapour temperature at the 8th effect",...
%             "Vapour temperature at the 10th effect",...
%             "Vapour temperature at the 12th effect",...
%             "Vapour temperature at the 14th effect"
%         ];

        %% Steady state parameters
        % Variables to check for the steady state operation of the plant
        steadyState_vars =      ["Ts_in", "Ts_out", "Ms", "Tcwin", "Tcwout", "Tf", "Mf", "Msw", "Tvc", "Mprod"];
        % Max allowed variation from the mean for each of the steady state
        % variables
        steadyState_maxAllVar = 2e-2; % 2% by default
        % Time where each steady state variable must remain within the
        % limits
        steadyStateTime = 600;
        % Units of steadyStateTime
        steadyStateUnit = "seconds";
        
        %% Plant parameters
        % Number of effects of the MED plant
        Nef = 14;
        % Area of the condenser in m2
        Ac = 18.3;
        % Area of the first effect in m2
        Aef_1 = 24.26;
        % Area of the effect N in m2
        Aef = 26.28;
        % Area of the preheater N in m2
        Aph = 5;
        % Boolean variable to include or not mixers in the model
        MIXER = true;
        % Proportion of mass flow in the mixers
        Y = [0.1, 0.5, 0.5];
        % Effect with flow metter
        ef_inst = 14;   % Could be an array
        % Phase of the heat source at the first effect
        source = "liquid";
        % Default salinity for the feedwater in g/kg
        Xf = 3306.24/1000;

        %% Processed point properties


        %% Database properties (MongoDB)
        server = "localhost";
        port = 27017;
        dbname = "test_db";
        collection = "operationPoints3";
        user = 'test_user';
        password = 'test_user';
    end
    
    methods
        function ptOp = generateOpPtObj(obj, nomenclatura)
        % Method to generate a new data type "Operation point". It is
        % a structure with different fields to include all the
        % necessary variables for the caracterization of a MED plant
        %
        % Input "nomenclatura" must be either "old" or "new"
        %
        % ptOp, struct with fields:
        %                 time: datetime
        %                  Nef: int
        %                   Ac: double
        %                  Aef: [14×1 double]
        %                  Aph: [14×1 double]
        %                MIXER: bool
        %                    Y: [3x1 double]
        %              ef_inst: int
        %               source: "liquid" or "steam"
        %     steadyState_time: [1×1 struct]
        %              medidas: [1×1 struct]
        %          steadyState: [1×1 struct]
        %                notes: string
        %
        % where medidas:
        %          Ts_in: [1×1 struct]
        %         Ts_out: [1×1 struct]
        %             Tf: [1×1 struct]
        %             Mf: [1×1 struct]
        %            Msw: [1×1 struct]
        %             Ms: [1×1 struct]
        %          Mprod: [1×1 struct]
        %          Tprod: [1×1 struct]
        %            Tvc: [1×1 struct]
        %          Tcwin: [1×1 struct]
        %         Tcwout: [1×1 struct]
        %             Ps: [1×1 struct]
        %     Tph_ref_13: [1×1 struct]
        %       Tv_ref_1: [1×1 struct]
        %       Tv_ref_2: [1×1 struct]
        %       Tv_ref_4: [1×1 struct]
        %       Tv_ref_6: [1×1 struct]
        %       Tv_ref_8: [1×1 struct]
        %      Tv_ref_10: [1×1 struct]
        %      Tv_ref_12: [1×1 struct]
        %      Tv_ref_14: [1×1 struct]
        %             Xf: [1×1 struct]
        %
        % where Ts_in, struct with fields:
        %       sensor_id: string (i.e.: "TT-AQU-107a")
        %          var_id: string (i.e.: "Ts_in")
        %     description: string (i.e.: "Inlet temperature of the
        %     source at the first effect")
        %            unit: string (i.e.: "ºC")
        %           value: double
        %         rawData: [ptOp.steadyState_time.Valuex1 double]
        %
        % where steadyState, struct with similar fields to medidas:
        % steadyState.variable, struct with fields:
        %            var_id: string (i.e.: "Ts_in")
        %     maxAllowedVar: double (i.e.: 10e-2 (10%))
        %    maxMeasuredVar: double (i.e.: 6e-2 (<maxAllowedVar))

        if strcmp(nomenclatura, "old")
            sensor_ids = obj.sensor_ids_antiguo; 
        elseif strcmp(nomenclatura, "new")
            sensor_ids = obj.sensor_ids_nuevo;
        elseif strcmp(nomenclatura, "2022")
            sensor_ids = obj.sensor_ids_2022;
        else
            error('Value must be either "new" or "old" or "2022" for nomenclatura')
        end

        medidas = [];
        steadyState = [];
        steadyState_param = struct('Value', obj.steadyStateTime, ...
            'Unit', obj.steadyStateUnit);
        
        for i=1:length(obj.var_ids)
            if sensor_ids(i) ~= "-" 
                medidas.(obj.var_ids(i)) = struct(...
                    'sensor_id', sensor_ids(i), ...
                    'var_id', obj.var_ids(i),...
                    'description', obj.descriptions(i),...
                    'unit', obj.units_SCADA(i),...
                    'unit_SCADA', obj.units_SCADA(i),...
                    'unit_model', obj.units_model(i),...
                    'value', [],...
                    'rawData', [] );
            end
        end
        medidas.Xf = struct(...
                'sensor_id', 'null', ...
                'var_id', "Xf",...
                'description', "Feedwater salinity",...
                'unit', "ppm",...
                'unit_SCADA', "ppm",...
                'unit_model', "g/kg",...
                'value', obj.Xf,...
                'rawData', nan );

        for i=1:length(obj.steadyState_vars)
           steadyState.(obj.steadyState_vars(i)) = struct(...
                'var_id', obj.steadyState_vars(i),...
                'maxAllowedVar', obj.steadyState_maxAllVar,...
                'maxMeasuredVar',[] ); 
        end
        steadyState.time = obj.steadyStateTime;

        ptOp = struct(...
            'time',NaT,...
            'Nef', obj.Nef, ...
            'Ac', obj.Ac, ...
            'Aef', [24.26; 26.28*ones(obj.Nef-1,1)], ...
            'Aph', [5*ones(obj.Nef-1,1); 18.3],...
            'MIXER', obj.MIXER,...
            'Y', obj.Y,...
            'ef_inst',  obj.ef_inst,...
            'source', obj.source,...
            'steadyState_time', steadyState_param,...
            'medidas', medidas,....
            'steadyState', steadyState,...
            'notes', ""...
        );
    end
        
        
    function toDatabase(obj, punto_op, varargin)
        % Función para exportar un objeto punto de operación (uno con
        % la misma estructura que la generada por generateOpPtObj) a
        % una base de datos MongoDB
        %
        % Method to export an object (or cell array of objects) of type 
        % "Operation point" (like the one generated by this class method 
        % generateOpPtObj) to a MongoDB database 
        %
        % Use:
        % count = toDatabase(obj, punto_op)

        % Import java libraries: jdk8 (java.text.SimpleDateFormat)
        %                        bson (bson.types.*, bson.Dcoument)
        %                        mongdb (mongodb.*)

        iP = inputParser;
%         addParameter(iP, 'uoda', "all", @(x)validateattributes(x, {'string'},{'vector'}))
        addParameter(iP, 'update', false, @(x)validateattributes(x, {'logical'},{'scalar'}))
        parse(iP,varargin{:})

%         fields_to_import = iP.Results.fields;
        update = iP.Results.update;
        
        % Import JAVA libraries if not already imported
        try
           com.mongodb.client.MongoClients.create;
        catch ME
            if strcmp(ME.identifier, 'MATLAB:dispatcher:noMatchingConstructor') || ...
               strcmp(ME.identifier, 'MATLAB:undefinedVarOrClass')
                obj.importLibraries;
            end
        end

        % Establish connection to database using JAVA library and
        % publish new entry
        if isempty(obj.user)
            mongoClient = com.mongodb.client.MongoClients.create(...
                sprintf("mongodb://%s:%d", obj.server, obj.port));
        else
            connectionURL = sprintf('mongodb+srv://%s:%s@medpsa.myhxe.mongodb.net/myFirstDatabase?retryWrites=true&w=majority', ...
            obj.user, obj.password);
%             connectionURL = sprintf("mongodb+srv://%s:%s@medpsa.myhxe.mongodb.net/?authSource=admin&replicaSet=atlas-qx88my-shard-0&readPreference=primary&ssl=true", ...
%                 obj.user, obj.password);
            disp(connectionURL)
            mongoClient = com.mongodb.client.MongoClients.create(connectionURL);
        end
        
        db =  mongoClient.getDatabase(obj.dbname);
        col = db.getCollection(obj.collection);

        % Set up JAVA datetime object
        ft = java.text.SimpleDateFormat("dd-MMM-yyyy HH:mm:ss");
        ft.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));

        % If update document with calibration data
        if update
            original_pt = obj.fromDatabase(single_document = punto_op.time);
            % Save old calibrations
            if isfield(original_pt, "processedData")
                if ~isfield(original_pt, "processedData_old")
                    original_pt.processedData_old = original_pt.processedData;
                else

                    % If dissimilar structures append empty missing fields to lacking structures
                    try
                        original_pt.processedData_old = [original_pt.processedData_old; original_pt.processedData];
                    catch Error
                        % Add missing fields
                        if strcmp(Error.identifier, 'MATLAB:heterogeneousStrucAssignment')
                            missing_fields = setdiff( string(fieldnames(original_pt.processedData)), string(fieldnames(original_pt.processedData_old(1))) );
                            for mf_idx=1:length(missing_fields) 
                                for old_idx=1:length(original_pt.processedData_old)
                                    if ~isfield(original_pt.processedData_old(old_idx), missing_fields(mf_idx))
                                        original_pt.processedData_old(old_idx).(missing_fields(mf_idx)) = "";
                                    end
                                end
                            end
                        else
                            throw(Error)
                        end
                    end
                end
            end

            % Add new calibration
            original_pt.processedData = punto_op.processedData;
            punto_op = original_pt;
            % Add time of calibration
            punto_op.processedData.calibrationTime = datetime("now", "Format", "dd-MMM-yyyy HH:mm:ss", "TimeZone", "UTC");
        end

        % Convert duration vectors to double vector in operatedTime field
        for idx=1:length(punto_op)
            ptop = punto_op(idx);
            if isfield(ptop, "operatedTime")
                fns = fieldnames(ptop.operatedTime.effects); fns=string(fns');
                for fieldname=fns
                    % Retrieve field
                    fieldValues = vertcat(ptop.operatedTime.effects.(fieldname));
                    if isa(fieldValues, 'duration')
                        % Convert to double
                        fieldValues = hours(fieldValues);
                        % Assign field back
                        tmp = mat2cell(fieldValues, ones(1,length(fieldValues)));
                        [ptop.operatedTime.effects.(fieldname)] = deal(tmp{:});
                    end
    
                end
            end
            punto_op(idx) = ptop;
        end

        
        %% Data configuration

        % Single point
        
        if numel(punto_op) == 1
            % From MATLAB object to JSON string
            obj_json = jsonencode(punto_op);
        
            % From JSON string to mongodb object
            obj_json_mongo = org.bson.Document.parse(obj_json);
    
            % Time field
            if isfield(punto_op, "time")
                % Make sure the time is UTC, generate a warning if it was not
                % the case
                if ~strcmp(punto_op.time.TimeZone, "UTC")
                    punto_op.time.TimeZone="UTC";
                    warning('TimeZone property was set to "UTC" for field "time"')
                end
                punto_op.time.Format = "dd-MMM-uuuu HH:mm:ss";
        
                % Substitute "time" field for the proper JAVA format
                obj_json_mongo.remove("time");
                
                time = ft.parse(string( punto_op.time ));
                obj_json_mongo.put("time", time);
            end

            % Time field in processedData
%             if isfield(punto_op, "processedData")
%             if isfield(punto_op.processedData, "calibrationTime")
%                 % Make sure the time is UTC, generate a warning if it was not
%                 % the case
%                 if ~strcmp(punto_op.processedData.calibrationTime.TimeZone, "UTC")
%                     punto_op.processedData.calibrationTime.TimeZone="UTC";
%                     warning('TimeZone property was set to "UTC" for field "time"')
%                 end
%                 punto_op.processedData.calibrationTime.Format = "dd-MMM-uuuu HH:mm:ss";
%         
%                 % Substitute "time" field for the proper JAVA format
%                 obj_json_mongo.remove("processedData.calibrationTime");
%                 
%                 time = ft.parse(string( punto_op.time ));
%                 obj_json_mongo.put("processedData.calibrationTime", time);
%             end
%             end

        if ~update
            % Insert into database
            col.insertOne(obj_json_mongo);
        else
            % Update document in database
            query = org.bson.Document;
            query.put( "time", time );
            result = col.replaceOne(query, obj_json_mongo);

            if result.getModifiedCount()<1
                error('ptOp:toDatabase_update', 'Operation point with calibrated parameters non updated correctly')
            end
        end


        % Multiple points
        else
        % Make sure the time is UTC, generate a warning if it was not
        % the case

        for i=1:length(punto_op)
            % From MATLAB object to JSON string
            obj_json = jsonencode(punto_op(i));

            % From JSON string to mongodb object
            obj_json_mongo = org.bson.Document.parse(obj_json);

            if isfield(punto_op(i), "time")
                if punto_op(i).time.TimeZone ~= "UTC" %#ok<*BDSCI> 
                    punto_op(i).time.TimeZone="UTC";
                    punto_op(i).time.Format = "dd-MMM-uuuu HH:mm:ss";
                    warning('ptOp:toDatabase','TimeZone property was set to "UTC" for field "time" (ptop: %d)', i);
                end
        
                % Substitute "time" field for the proper JAVA format
                obj_json_mongo.remove("time");
                time = ft.parse(string( punto_op(i).time ));
                obj_json_mongo.put("time", time);        
            end
            
            if ~update
                % Insert into database
                col.insertOne(obj_json_mongo);
            else
                % Update document in database
                error('ptOp:toDatabase_update', 'Multiple documents update not supported')
                
            end
        end

        end


        % Inform if successfull
        fprintf("Operation point(s) exported succesfully to database (%s/%s)\n", ...
            obj.dbname, obj.collection);
        if update
            fprintf("Check updated document in MongoDB Compass with: { time: new ISODate('%s') }\n", ...
                    datestr(punto_op.time, 'yyyy-mm-ddTHH:MM:ss.FFF+00:00'))
        end
            mongoClient.close()
        end

        function datos = fromDatabase(obj, varargin)
        % Method to import "Operation Points" data from a MongoDB
        % database. "Operation point" data type is of the kind
        % generated by this class method generateOpPtObj
        %
        % Use:
        % datos = importFromDatabase()
        % 
        % This method takes no inputs and outputs the requested data.
        % The connection to the database is configured trough the
        % class properties server, port, dbname and collection
        iP = inputParser;
        addParameter(iP, 'fields', "all", @(x)validateattributes(x, {'string'},{'vector'}))
        addParameter(iP, 'sort', false, @(x)validateattributes(x, {'logical'},{'scalar'}))
        addParameter(iP, 'single_document', datetime(0,0,0, TimeZone="UTC"), @(x)validateattributes(x, {'datetime'},{'scalar'}))
        parse(iP,varargin{:})

        fields_to_import = iP.Results.fields;
        sort = iP.Results.sort;
        single_doc_datetime = iP.Results.single_document;

        if single_doc_datetime ~= datetime(0,0,0, TimeZone="UTC")
            % Set up JAVA datetime object
            ft = java.text.SimpleDateFormat("dd-MMM-yyyy HH:mm:ss");
            ft.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));
            time = ft.parse(string( single_doc_datetime ));

            find_content = org.bson.Document;
            find_content.put( "time", time );
        end

%         if fields_to_import ~= "all"
%             fields = struct();
%             for fname=fields_to_import
%                 fields.(fname) = 1.0;
%             end
%             fields = struct2cell(fields);
%         end

%         fields = '{"department":1.0,"salary":1.0}';
        
        % Import JAVA libraries if not already imported
        try
           com.mongodb.client.MongoClients.create;
        catch ME
            if strcmp(ME.identifier, 'MATLAB:dispatcher:noMatchingConstructor') || ...
               strcmp(ME.identifier, 'MATLAB:undefinedVarOrClass')

                obj.importLibraries;
            end
        end
        
        % Establish connection to database using JAVA library and
        % publish new entry
        % logger = org.apache.log4j.Logger.getLogger("com.mongodb.MongoClient");
        % org.apache.log4j.BasicConfigurator.configure();
        % logger.setLevel(org.apache.log4j.Level.INFO);

        data = [];
        
        if isempty(obj.user)
            mongoClient = com.mongodb.client.MongoClients.create(...
                sprintf("mongodb://%s:%d", obj.server, obj.port));
        else
            connectionString = sprintf('mongodb+srv://%s:%s@medpsa.myhxe.mongodb.net/myFirstDatabase?retryWrites=true&w=majority', ...
                obj.user, obj.password);

%             connectionString = sprintf("mongodb+srv://%s:%s@medpsa.myhxe.mongodb.net/?authSource=admin&replicaSet=atlas-qx88my-shard-0&readPreference=primary&ssl=true", ...
%                 obj.user, obj.password);
            mongoClient = com.mongodb.client.MongoClients.create(connectionString);
        end
        
        db =  mongoClient.getDatabase(obj.dbname);
        col = db.getCollection(obj.collection);
        
%         if fields_to_import == "all"
            if sort
                % Single document sorted
                if single_doc_datetime ~= datetime(0,0,0, TimeZone="UTC")
                    cursor = col.find(find_content).sort(com.mongodb.client.model.Sorts.ascending("time")).iterator();
                % Whole database sorted
                else
                    cursor = col.find().sort(com.mongodb.client.model.Sorts.ascending("time")).iterator();
                end
            else
                % Single document non sorted
                if single_doc_datetime ~= datetime(0,0,0, TimeZone="UTC")
                    cursor = col.find(find_content).iterator();
                % Whole database non sorted
                else
                    cursor = col.find().iterator();
                end
            end
%         else
%             fields_to_not_import = setdiff(string(fieldnames()), fields_to_import);
%             if ~any(contains(fields_to_import, "time"))
%                 fields_to_import = [fields_to_import "time"];
%             end
%             variables_to_import = strcat("medidas.", fields_to_import);
%             fields_to_import = [fields_to_import string(fieldnames(pun))']
%             fields = com.mongodb.client.model.Projections.include(fields_to_import);
%             projection = com.mongodb.client.model.Projections.fields(fields);
%             Convert to bson
%             fields = org.bson.Document(fields);
%             if sort
%                 cursor = col.find().projection(fields).sort(com.mongodb.client.model.Sorts.ascending("time")).iterator();
%                 
%             else
%                 cursor = col.find().projection(fields).iterator();
%             end
%         end

        input_formats = ["dd-MMM-uuuu", "dd/MM/uuuu"]; input_format = input_formats(1);
        input_formats_with_time = ["dd-MMM-uuuu HH:mm:ss", "dd/MM/uuuu HH:mm:ss"];
        input_format_with_time = input_formats_with_time(1);
        i=1;
        try 
            while (cursor.hasNext()) 
                data_json = string( cursor.next().toJson() );
                data = jsondecode(data_json);
                if isfield(data, "time")
                    data.time = datetime(...
                        data.time.x_date, ...
                        InputFormat = "uuuu-MM-dd'T'HH:mm:ss.SSS'Z'", ...
                        Timezone = "UTC" );
                end
                
                % Eliminar campo x_ids
                fns = string(fieldnames(data))';
                for x_idx= find(contains(fns, 'x_id'))
%                     disp(fns(x_idx))
                    data = rmfield(data,fns(x_idx)); 
                end
                % Filter fields
                if fields_to_import ~= "all"
                    fields_to_not_import = setdiff(string(fieldnames(data.medidas)), fields_to_import);
                    data.medidas = rmfield(data.medidas,fields_to_not_import);
                end

                % Convert double vectors to duration vectors in operatedTime field
                % as well as: lastOperationTime, lastCleanupDate to datetimes
                if isfield(data, "operatedTime")
                    fns = fieldnames(data.operatedTime.effects); fns=string(fns');
                    for fieldname=fns
                        % effects

                        % Retrieve field
                        fieldValues = vertcat( [data.operatedTime.effects.(fieldname)]' );
                        % Convert to duration
                        fieldValues = hours(fieldValues);
                        % Assign field back
                        tmp = mat2cell(fieldValues, ones(1,length(fieldValues)));
                        [data.operatedTime.effects.(fieldname)] = deal(tmp{:});
                        
                        try
                            % lastOperationTime
                            data.operatedTime.lastCleanupDate   = datetime(data.operatedTime.lastCleanupDate, 'InputFormat', input_format);
                            % lastOperationTime
                            data.operatedTime.lastOperationTime = datetime(data.operatedTime.lastOperationTime, 'InputFormat', input_format_with_time);
                        catch ME 
                            if contains( ME.identifier, 'MATLAB:datetime:ParseErr' )
                                input_format = setdiff(input_formats, input_format); input_format = input_format(1);
                                input_format_with_time = setdiff(input_formats_with_time, input_format_with_time); 
                                input_format_with_time = input_format_with_time(1);
                            else
                                throw(ME);
                            end

                        end
                            
                    end
                end


                % Append processed document to list of documents
                data_aux = data;
                % There might be a mix of processed and non-processed
                % points, add empty field to non processed to avoid
                % disimilar structures error
                if ~isfield(data, "processedData"), data_aux.processedData = struct; end
                if ~isfield(data_aux.processedData, "calibrationTime")
                    data_aux.processedData.calibrationTime = ""; 
                    data_aux.processedData_old = [];
                end
                datos(i) = data_aux; %#ok<*AGROW> 
                i=i+1;
            end
        catch MException
            cursor.close();
            throw(MException)
        end
        
        if ~isempty(data)
            if numel(datos)>1
                if isfield(data, "time")
                    datos = table2timetable( struct2table(datos) );
                end
                fprintf('Imported <strong>%d</strong> operation points from datatabase\n', height(datos))
            end
        else
            warning('ptOp:fromDatabase', 'No data retrieved from database');
            datos = [];
        end
        
            mongoClient.close();
        end
        
        function importLibraries(~)
        % Method used by other methods of this class to import the
        % necessary jar files for the libraries used
        
        currentDirectory = pwd;
        librariesDirectory = fullfile(currentDirectory, "lib");
        fileinfo = dir(librariesDirectory);
        if isempty(fileinfo)
            warning('Could not find "lib" folder in %s', librariesDirectory);
        else
            files = {fileinfo.name};
            for i=1:length(files)
                if ~strcmp(files{i}(1), '.') && contains(files{i}, ".jar")
                    javaaddpath(fullfile( librariesDirectory, files{i} ));
                    fprintf('Imported %s to java path\n', files{i});
                end
            end
            % End of for
        end
        % End of method
        end
       

    % END OF METHODS
    end

% END OF CLASS
end