import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker and set seeds for reproducibility
fake = Faker()
np.random.seed(42)
random.seed(42)

# Define parameters for dataset generation
num_lawyers = 200
min_cases_per_lawyer = 30
max_cases_per_lawyer = 300
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 5, 1)  

# Define Indian law schools and their tiers
indian_law_schools = {
    "Tier 1": [
        "National Law School of India University, Bangalore",
        "National Academy of Legal Studies and Research, Hyderabad",
        "National Law University, Delhi",
        "West Bengal National University of Juridical Sciences, Kolkata",
        "National Law University, Jodhpur"
    ],
    "Tier 2": [
        "Gujarat National Law University, Gandhinagar",
        "National Law University, Mumbai",
        "National Law University, Bhopal",
        "National Law University, Cuttack",
        "National Law University, Patna",
        "National Law University, Lucknow",
        "National Law University, Raipur",
        "National Law University, Ranchi",
        "National Law University, Assam",
        "National Law University, Punjab"
    ],
    "Tier 3": [
        "Symbiosis Law School, Pune",
        "ILS Law College, Pune",
        "Government Law College, Mumbai",
        "Campus Law Centre, Delhi University",
        "Law Centre-I, Delhi University",
        "Faculty of Law, Banaras Hindu University",
        "Faculty of Law, Aligarh Muslim University",
        "Faculty of Law, University of Calcutta",
        "Faculty of Law, University of Madras",
        "Faculty of Law, University of Mumbai"
    ]
}

# Define school tier factors - adding the missing function
def get_school_factors(tier):
    """Return performance factors based on school tier."""
    factors = {
        "Tier 1": {
            "prestige_factor": 1.3,
            "complex_case_probability": 0.7
        },
        "Tier 2": {
            "prestige_factor": 1.1,
            "complex_case_probability": 0.5
        },
        "Tier 3": {
            "prestige_factor": 0.9,
            "complex_case_probability": 0.3
        }
    }
    return factors[tier]

indian_case_types = {
    "Criminal": {
        "complexity": 1.3,
        "duration_range": (30, 365),
        "common_courts": ["District Court", "High Court", "Supreme Court"],
        "success_rate": 0.6
    },
    "Civil": {
        "complexity": 1.2,
        "duration_range": (90, 730),
        "common_courts": ["District Court", "High Court"],
        "success_rate": 0.55
    },
    "Family": {
        "complexity": 1.1,
        "duration_range": (60, 365),
        "common_courts": ["District Court", "High Court"],
        "success_rate": 0.5
    },
}

indian_court_levels = {
    "District Court": {
        "complexity_factor": 1.0,
        "jurisdiction": "Local",
        "appeal_to": "High Court",
        "case_types": ["Criminal", "Civil", "Family"]  
    },
    "High Court": {
        "complexity_factor": 1.5,
        "jurisdiction": "State",
        "appeal_to": "Supreme Court",
        "case_types": ["Criminal", "Civil", "Family"]
    },
    "Supreme Court": {
        "complexity_factor": 2.0,
        "jurisdiction": "National",
        "appeal_to": None,
        "case_types": ["Criminal", "Civil", "Family"] 
    },
}

indian_case_outcomes = {
    "Won": {
        "probability_factor": 1.0,
        "revenue_multiplier": 1.2,
        "description": "Case decided in favor of the client"
    },
    "Lost": {
        "probability_factor": 0.0,
        "revenue_multiplier": 0.8,
        "description": "Case decided against the client"
    },
    "Settled": {
        "probability_factor": 0.7,
        "revenue_multiplier": 1.0,
        "description": "Case resolved through mutual agreement"
    }
}


def random_date(start, end):
    """Generate a random date between start and end dates."""
    time_between = end - start
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start + timedelta(days=random_days)


def get_random_law_school():
    """Get a random law school with its tier."""
    tier = random.choices(
        ["Tier 1", "Tier 2", "Tier 3"],
        weights=[0.1, 0.3, 0.6],  # Fewer Tier 1 schools, more Tier 3
        k=1
    )[0]
    school = random.choice(indian_law_schools[tier])
    return school, tier

def generate_lawyer_name():
    """Generate a realistic Indian lawyer name using Faker."""
    # Set locale to Indian English for Indian names
    fake = Faker('en_IN')
    return fake.name()

def generate_lawyer_data(num_lawyers=200):  # Changed default to match the parameter
    """Generate synthetic lawyer data with Indian context."""
    lawyers = []
    
    for lawyer_id in range(1, num_lawyers + 1):
        # Generate dates
        start_practice = random_date(start_date, end_date - timedelta(days=365*2))
        years_exp = (end_date - start_practice).days / 365.25
        
        # Get law school and its tier
        law_school, school_tier = get_random_law_school()
        
        # Get school factors - Need to be accessed here
        school_factors = get_school_factors(school_tier)
        
        # Generate lawyer profile
        lawyer = {
            'lawyer_id': lawyer_id,
            'name': generate_lawyer_name(),
            'law_school': law_school,
            'school_tier': school_tier,
            'start_practice': start_practice.strftime('%Y-%m-%d'),
            'years_experience': round(years_exp, 1),
        }
        
        # Generate case type frequencies with realistic distribution
        total = 100
        frequencies = []
        remaining = total
        
        # Sort case types by complexity
        sorted_case_types = sorted(indian_case_types.items(), 
                                 key=lambda x: x[1]['complexity'], 
                                 reverse=True)
        
        for i, (case_type, details) in enumerate(sorted_case_types[:-1]):
            # More experienced lawyers handle more complex cases
            complexity_factor = details['complexity']
            exp_factor = min(1.0, years_exp / 10)
            school_case_factor = school_factors["complex_case_probability"]
            
            # Base frequency adjusted by complexity, experience, and school tier
            base_freq = 5 + (complexity_factor * 10 * exp_factor * school_case_factor)
            max_freq = remaining - (len(sorted_case_types) - i - 1) * 5
            freq = min(int(base_freq), max_freq)
            
            frequencies.append(freq)
            remaining -= freq
        
        frequencies.append(remaining)
        lawyers.append(lawyer)
    
    return pd.DataFrame(lawyers)

def generate_case_data(lawyer_df):
    """Generate synthetic case data with Indian context."""
    cases = []
    case_id = 1
    
    for _, lawyer in lawyer_df.iterrows():
        start_practice_date = datetime.strptime(lawyer['start_practice'], '%Y-%m-%d')
        years_exp = lawyer['years_experience']
        school_factors = get_school_factors(lawyer['school_tier'])
        
        # Calculate number of cases based on experience and school tier
        experience_factor = min(1.0, years_exp / 10)
        school_case_factor = school_factors["complex_case_probability"]
        num_cases = int(min_cases_per_lawyer + (max_cases_per_lawyer - min_cases_per_lawyer) * 
                       experience_factor * school_case_factor)
        
        for _ in range(num_cases):
            # Higher chance of complex cases for higher tier schools
            if random.random() < school_factors["complex_case_probability"]:
                case_type_candidates = [ct for ct, details in indian_case_types.items() 
                                       if details['complexity'] > 1.2]
                if case_type_candidates:
                    case_type = random.choice(case_type_candidates)
                else:
                    case_type = random.choice(list(indian_case_types.keys()))
            else:
                case_type = random.choice(list(indian_case_types.keys()))
            
            # Get case type details
            case_details = indian_case_types[case_type]
            
            # Case start date
            case_start_date = random_date(start_practice_date, end_date - timedelta(days=30))
            
            # Determine court level based on case type and complexity
            eligible_courts = [court for court, details in indian_court_levels.items() 
                             if case_type in details['case_types']]
            
            if not eligible_courts:  # Fallback if no courts are eligible
                eligible_courts = list(indian_court_levels.keys())
                
            court_weights = [indian_court_levels[level]['complexity_factor'] for level in eligible_courts]
            court_level = random.choices(eligible_courts, weights=court_weights, k=1)[0]
            
            # Calculate case duration
            min_duration, max_duration = case_details['duration_range']
            complexity_factor = case_details['complexity'] * indian_court_levels[court_level]['complexity_factor']
            base_duration = random.randint(min_duration, max_duration)
            case_duration = int(base_duration * complexity_factor)
            case_end_date = min(case_start_date + timedelta(days=case_duration), end_date)
            
            # Determine case outcome with school tier and case type influence
            base_success_rate = case_details['success_rate']
            success_probability = base_success_rate  # Start with base success rate
            
            # Adjust for complexity
            if complexity_factor > 1.3:
                success_probability *= 0.9  # Harder to win complex cases
                
            # Adjust for school prestige    
            success_probability *= school_factors["prestige_factor"]
            
            # Select outcome based on probability
            if random.random() < success_probability:
                # Higher chance of winning for better lawyers
                if random.random() < 0.7:  # 70% chance of winning if successful
                    outcome = "Won"
                else:
                    outcome = "Settled"
            else:
                outcome = "Lost"
            
            case = {
                'case_id': case_id,
                'lawyer_id': lawyer['lawyer_id'],
                'case_type': case_type,
                'court_level': court_level,
                'start_date': case_start_date.strftime('%Y-%m-%d'),
                'end_date': case_end_date.strftime('%Y-%m-%d'),
                'duration_days': (case_end_date - case_start_date).days,
                'result': outcome,
            }
            cases.append(case)
            case_id += 1
    
    return pd.DataFrame(cases)

def save_data(lawyer_df, case_df):
    """Save generated data to CSV files."""
    lawyer_df.to_csv('data/lawyers.csv', index=False)
    case_df.to_csv('data/cases.csv', index=False)

def main():
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    lawyer_df = generate_lawyer_data(num_lawyers)  # Pass the parameter
    case_df = generate_case_data(lawyer_df)
    
    # Save data
    save_data(lawyer_df, case_df)
    print("Data generation complete. Files saved in data/ directory.")
    
    # Print data preview
    print("\nLawyer Data Preview:")
    print(lawyer_df.head(3).to_string())
    print("\nCase Data Preview:")
    print(case_df.head(3).to_string())

if __name__ == "__main__":
    main()