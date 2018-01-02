function [str] = struct2str(s, style)
    % Codes for converting struct to string
    % 
    % ARGS:
    % s        : the given structure
    % style    : the style we want (table or a single-line string)
    %
    % RETURN:
    % str      : the resulted string

    names=fieldnames(s);
    values=struct2cell(s);

    str = '';
    if strcmp(style, 'table')
        for i = 1:length(names)
            str = sprintf('%s%s:\t%s\n', str, names{i}, num2str(values{i}));
        end
    elseif strcmp(style, 'string')
        for i = 1:length(names)
            str = sprintf('%s%s=%s_', str, names{i}, num2str(values{i}));
        end
        str = str(1:end-1);
    end

end