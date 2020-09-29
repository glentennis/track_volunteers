#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import datetime
import re
import os
import gspread
import yaml

settings = yaml.load('settings.yaml')

CLIENT_SECRETS_PATH = settings['client_secrets_file']
SHEET_ID = settings['output_sheet_id']
SOURCE_DATA_DIR = settings['source_data_dir']

SOURCE_FILES = {
    'contacts': [
        "Campaign Contact List - Full List.csv",
    ]
    ,'shifts':[
        "Actual Shifts 7.27-8.27 - ActualShifts.csv",
        "Actual Shifts 0827 to 0908.csv",
        "Actual Shifts 0910 to 0915.csv",
        'Actual Shifts 9_16 - 9_22 - Sheet1 (1).csv',
    ]
    ,'signups':[
        "Sign Ups for SSC (Sunday_s).csv",
        "Remaining August Sign Ups.csv",
    ]
}


def get_files(key):
    return [os.path.join(SOURCE_DATA_DIR, f) for f in SOURCE_FILES[key]]


def parse_action_date(x):
    if isinstance(x, str):
        x = re.findall("[0-9]{1,2}(?:\\.|\\/)[0-9]{1,2}", x)[0]
        x = re.sub('\\.', '/', x)
        x = x + '/2020'
        x = datetime.datetime.strptime(x, '%m/%d/%Y').strftime('%Y-%m-%d')
    return x


def strip_phone_number(x):
    if isinstance(x, str):
        x = re.sub("[^0-9]", "", x)
        x = x[-10:]
    return x


def get_full_name(first, last):
    x = str(first) + ' ' + str(last)
    return x.lower()


def slugify(x):
    return x.replace(' ', '_').lower()


def get_mapping(primary_df, primary_key, foreign_df, foreign_key, left_criteria=[], right_criteria=[]):
    """
    Finds approximate matches between primary_key 
        in primary_df and foreign_key in foreign_df,
        based on criteria fields. Assumes that there
        should only be one foreign_key match per primary_key.
    
    Input:
        - primary_df dataframe
        - primary_key str
        - foreign_df dataframe
        - foreign_key str
        - left_criteria list: A list of fields from primary_df that will be used 
            to match with foreign_df. Fields are listed in order of priority,
            meaning matches from the first field in the list will take precedence
            over the second field in the list, and so on.
        - right_criteria list: Corresponing fields from foreign_df

    
    """
    mapping_dfs = []
    for i, key in enumerate(left_criteria):
        merged = pd.merge(
            primary_df.dropna(subset=[left_criteria[i]]),
            foreign_df.dropna(subset=[right_criteria[i]]),
            left_on=left_criteria[i],
            right_on=right_criteria[i],
        )
        merged = merged[[primary_key, foreign_key]]
        mapping_dfs.append(merged)
    mappings = pd.concat(mapping_dfs, axis=0)
    mappings = mappings.drop_duplicates(
        subset=[primary_key], # one foreign_key per primary_key
        keep='first', # takes first match
    )
    return mappings


def get_mapped_df(primary_df, primary_key, foreign_df, foreign_key, left_criteria=[], right_criteria=[]):
    mappings = get_mapping(primary_df, primary_key, foreign_df, foreign_key, left_criteria, right_criteria)
    mapped = pd.merge(
        primary_df,
        mappings,
        on=primary_key
    )
    return mapped


def funnelize(df, action_id, contact_id, action_date, record_type):
    df = df.copy()
    df['action_id'] = df[action_id]
    df['contact_id'] = df[contact_id]
    df['action_date'] = df[action_date]
    df['record_type'] = record_type
    df = df[['action_id', 'contact_id', 'action_date', 'record_type']]
    return df


def process_shifts(df):
    df = df.replace(r'^\s*$', np.nan, regex=True) # replace blanks with NULL
    cols = ['Full Name', 'Date', 'Contact', 'Email']
    for c in cols:
        if c not in df.columns.values:
            df[c] = None
    df = df[cols]
    df.columns = ['full_name', 'shift_date', 'phone', 'email']
    df['full_name'] = df.full_name.apply(lambda x: str(x).lower())
    df['phone'] = df['phone'].apply(strip_phone_number)
    df['shift_date'] = df['shift_date'].apply(parse_action_date)
    return df


def process_signups(df):
    df = df.replace(r'^\s*$', np.nan, regex=True) # replace blanks with NULL
    df = df.melt(
        id_vars = ['First name', 'Last name', 'Group Affiliation']
    )
    mask = df.variable=="Sunday's for Systemic Change"
    df.loc[mask, 'variable'] = df.loc[mask, 'value']
    df = df[~df.value.isnull()]
    df = df[['First name', 'Last name', 'variable']]
    df.columns = ['first_name', 'last_name', 'action']
    df['signup_date'] = df['action'].apply(parse_action_date)
    return df


def process_contacts(df):
    df['phone'] = df['mobile_phone'].apply(strip_phone_number)
    df['full_name'] = df.apply(lambda row: get_full_name(row['first_name'], row['last_name']), axis=1)
    df['contact_id'] = np.arange(len(df))+1 # generate unique IDs
    df = df[["contact_id", "full_name", "phone","email", "join_date", "city", "branch", 'first_name', 'last_name']]
    return df


def get_data():
    contacts = [pd.read_csv(p) for p in get_files('contacts')][0] # just one file for now
    contacts = process_contacts(contacts)

    # since we're joining by name for signups, we have to dedupe names
    contacts_deduped_by_name = contacts.sort_values('join_date')
    contacts_deduped_by_name = contacts_deduped_by_name.drop_duplicates(
        'full_name', 
        keep='last'
    ) # take most recent member

    
    shift_dfs = [pd.read_csv(p) for p in get_files('shifts')]
    shift_dfs = [process_shifts(df) for df in shift_dfs]
    shifts = pd.concat(shift_dfs, axis=0)
    shifts['shift_id'] = np.arange(len(shifts))+1 # generate unique IDs

    signup_dfs = [pd.read_csv(p) for p in get_files('signups')]
    signup_dfs = [process_signups(df) for df in signup_dfs]
    signups = pd.concat(signup_dfs, axis=0)
    signups['signup_id'] = np.arange(len(signups))+1 # generate unique IDs
    signups['full_name'] = signups.apply(lambda row: get_full_name(row['first_name'], row['last_name']), axis=1)
    
    return contacts, contacts_deduped_by_name, shifts, signups


def validate_funnel(funnel):
    summary = funnel.pivot_table(
        index=['action_date','contact_id'],
        columns=['record_type'],
        values=['action_id'],
        aggfunc=pd.Series.nunique
    )
    duped_records = summary[summary>1].dropna(how='all')
    if len(duped_records)>1:
        raise Exception('There are contacts with multiple actions on the same day:\n\n{}'.format(duped_records))


def df_to_sheet(df, sheet):
    df = df.fillna('NA')
    sheet.update([df.columns.values.tolist()] + df.values.tolist())


if __name__ == '__main__':
    contacts, contacts_deduped_by_name, shifts, signups = get_data()

    ### MAP TO CONTACT IDS
    shifts_with_contacts = get_mapped_df( 
        shifts, 
        'shift_id',
        contacts, 
        'contact_id',
        left_criteria=['phone', 'email', 'full_name'],
        right_criteria=['phone','email', 'full_name']
    )
    signups_with_contacts = get_mapped_df(
        signups,
        'signup_id',
        contacts_deduped_by_name,
        'contact_id',
        left_criteria=['full_name'],
        right_criteria=['full_name']
    )

    ### BUILD FUNNEL
    funnelized_dfs = [
        funnelize(contacts, 'contact_id', 'contact_id', 'join_date', 'contact_created'),
        funnelize(shifts_with_contacts, 'shift_id', 'contact_id', 'shift_date', 'shift'),
        funnelize(signups_with_contacts, 'signup_id', 'contact_id', 'signup_date', 'signup'),
    ]
    funnel = pd.concat(funnelized_dfs, axis=0)
    funnel['week'] = pd.to_datetime(funnel['action_date']).dt.to_period('W-TUE').apply(lambda r: r.start_time).astype(str)
    funnel = pd.merge(
        funnel,
        contacts,
        on="contact_id"
    )
    cols = [
        'contact_id',
        'action_id',
        'action_date',
        'record_type',
        'first_name',
        'last_name',
        'city',
        'branch',
        'week'
    ]
    funnel = funnel[cols]
    validate_funnel(funnel)

    ### UPLOAD TO GOOGLE SHEETS
    gc = gspread.service_account(CLIENT_SECRETS_PATH)
    sheet = gc.open_by_key(SHEET_ID)
    
    df_to_sheet(funnel, sheet.worksheet("source-funnel"))
    df_to_sheet(contacts, sheet.worksheet("source-contacts"))
    df_to_sheet(signups, sheet.worksheet("source-signups"))
    df_to_sheet(shifts, sheet.worksheet("source-shifts"))
