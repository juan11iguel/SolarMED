function [pts, pts_original] = loadData(pts_original, varargin)

    iP = inputParser;
    check_logical = @(x)validateattributes(x, {'logical'},{'scalar'});
    check_string  = @(x)validateattributes(x, {'string'},{'scalar'});
    addParameter(iP, 'load_from_db', false, check_logical)
    addParameter(iP, 'filter_pts', true, check_logical)
    addParameter(iP, 'system', "MED", check_string)

    parse(iP,varargin{:})

    load_from_db = iP.Results.load_from_db;
    filter_pts = iP.Results.filter_pts;
    system = iP.Results.system;


    if load_from_db
        if strcmp(system, "MED")
        % Import data from database
        ptop_cls = ptOp;
    
        % Database parameters
        % When connecting to Atlas specify the user and password
        ptop_cls.user       = "test_user";
        ptop_cls.password   = "test_user";
        ptop_cls.dbname     = "MED_DB";
        ptop_cls.collection = "operation_points";
    
        variables_to_import = ["Ms"  "Mf" "Mprod" "Msw" "Ts_in" "Ts_out" ...
            "Tf" "Tprod" "Tvc" "Tcwin" "Tcwout"  "Tv_ref_1" "Tv_ref_2" ...
            "Tv_ref_4" "Tv_ref_6"  "Tv_ref_8" "Tv_ref_10" "Tv_ref_12" ...
            "Tv_ref_14" "Tph_ref_14"  "Ps" "Xf"];
        % Import from database
        pts = ptop_cls.fromDatabase(fields=variables_to_import);
        
        elseif strcmp(system, "WASCOP")
            ptop_cls = ptOp_wascop;
            % Database parameters
            % When connecting to Atlas specify the user and password
            ptop_cls.user       = "test_user";
            ptop_cls.password   = "test_user";
            ptop_cls.dbname     = "WASCOP";
            ptop_cls.collection = "operation_points";
        
            % Import from database
            pts = ptop_cls.fromDatabase();


        else
            throw(MException('loadData:unkownSystem', 'Unkown value for system, options are: MED, WASCOP'))
        end

        if ~issorted([pts.time]), pts = sortrows(pts); end
        % Convert to struct
        pts = table2struct( timetable2table(pts) );
    
        pts_original = pts;
        
    
    else
    
        % Load from workspace
    %     load pts_ann.mat
        pts = pts_original;
    
    end
    
    if strcmp(system, "MED")
    % Retrieve calibrated points
    processed_pts = [];
    for i=1:length(pts)
        if isfield(pts(i).processedData, "Uef")
            processed_pts = [processed_pts; pts(i)]; %#ok<*AGROW>
        end
    end
    pts = processed_pts; clear processed_pts

%     elseif strcmp(system, "WASCOP")
%     % Retrieve calibrated points
%     pts = []
    end
    
    %% Filter
    if filter_pts
        if strcmp(system, "MED")
            % Filter out invalid points 
            % (Mprod<0.2)
            pts = pts( [vertcat( vertcat(pts.medidas).Mprod ).value] > 1.6 );
            % (Tf>Tv(1))
            pts = pts( [vertcat( vertcat(pts.medidas).Tf ).value]' < tSatW([vertcat( vertcat(pts.medidas).Tv_ref_1 ).value]*1e-3) );
        
            pts = filterPts(pts, 'gfc', false);

        elseif strcmp(system, "WASCOP")
            % Filter out invalid points 


        end
    end