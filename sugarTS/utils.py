
def get_datetime(df, var_name="EventDateTime", targ_name="dt"):
    """
    Parse the datetime strings coming from the clarity and tandem exports.
    Replace the column of strings with a column of datetime objects

    Inputs:
    * df: dataframe
    * var_name: the name of the variable containing the datetime string
    * targ_name: the new name for the column of transformed datetime objects
    Outputs:
    * df: dataframe with replaced datetime column
    """
    df[var_name] = df[var_name].apply(lambda x: x[:19])
    df[targ_name] = df.apply(
        lambda row: datetime.datetime.strptime(
            row[var_name], "%Y-%m-%dT%H:%M:%S"
        ),
        axis=1,
    )
    df = df.drop([var_name], axis=1)
    return df


