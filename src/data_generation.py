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
end_date = datetime(2024, 4, 24)  # Current date

# Indian-specific data
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

# School tier factors
school_tier_factors = {
    "Tier 1": {
        "skill_bonus": 2.0,
        "starting_salary_multiplier": 1.5,
        "prestige_factor": 1.5,
        "complex_case_probability": 0.7
    },
    "Tier 2": {
        "skill_bonus": 1.0,
        "starting_salary_multiplier": 1.2,
        "prestige_factor": 1.2,
        "complex_case_probability": 0.5
    },
    "Tier 3": {
        "skill_bonus": 0.5,
        "starting_salary_multiplier": 1.0,
        "prestige_factor": 1.0,
        "complex_case_probability": 0.3
    }
}

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
        "common_courts": ["Family Court", "High Court"],
        "success_rate": 0.5
    },
    "Consumer": {
        "complexity": 1.0,
        "duration_range": (60, 180),
        "common_courts": ["Consumer Court", "State Commission"],
        "success_rate": 0.65
    },
    "Labour": {
        "complexity": 1.2,
        "duration_range": (90, 365),
        "common_courts": ["Labour Court", "Industrial Tribunal"],
        "success_rate": 0.55
    }
}

indian_court_levels = {
    "District Court": {
        "complexity_factor": 1.0,
        "jurisdiction": "Local",
        "appeal_to": "High Court",
        "case_types": ["Criminal", "Civil"]
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
        "case_types": ["Criminal", "Civil"]
    },
    "Family Court": {
        "complexity_factor": 1.2,
        "jurisdiction": "Family",
        "appeal_to": "High Court",
        "case_types": ["Family"]
    },
    "Consumer Court": {
        "complexity_factor": 1.1,
        "jurisdiction": "Consumer",
        "appeal_to": "State Commission",
        "case_types": ["Consumer"]
    },
    "Labour Court": {
        "complexity_factor": 1.2,
        "jurisdiction": "Labour",
        "appeal_to": "Industrial Tribunal",
        "case_types": ["Labour"]
    }
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

indian_awards = {
    "Senior Advocate Designation": {"prestige": 10, "min_experience": 10},
    "Padma Bhushan": {"prestige": 9, "min_experience": 20},
    "Padma Shri": {"prestige": 8, "min_experience": 15},
    "National Legal Services Authority Award": {"prestige": 7, "min_experience": 5},
    "State Legal Services Authority Award": {"prestige": 6, "min_experience": 5},
    "Bar Council of India Award": {"prestige": 7, "min_experience": 8},
    "State Bar Council Award": {"prestige": 6, "min_experience": 5},
    "Best Lawyer Award": {"prestige": 5, "min_experience": 3},
    "Pro Bono Excellence Award": {"prestige": 6, "min_experience": 3},
    "Young Lawyer of the Year": {"prestige": 4, "min_experience": 1}
}

def random_date(start, end):
    """Generate a random date between start and end dates."""
    time_between = end - start
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start + timedelta(days=random_days)

def get_awards_based_on_experience(years_exp):
    """Get awards based on experience and prestige."""
    eligible_awards = []
    for award, criteria in indian_awards.items():
        if years_exp >= criteria["min_experience"]:
            eligible_awards.append((award, criteria["prestige"]))
    
    if not eligible_awards:
        return []
    
    # Sort by prestige
    eligible_awards.sort(key=lambda x: x[1], reverse=True)
    
    # Select awards with probability based on prestige
    selected_awards = []
    for award, prestige in eligible_awards:
        if random.random() < (prestige / 10) * 0.5:
            selected_awards.append(award)
            if len(selected_awards) >= 4:  # Maximum 4 awards
                break
    
    return selected_awards

def get_random_law_school():
    """Get a random law school with its tier."""
    tier = random.choices(
        ["Tier 1", "Tier 2", "Tier 3"],
        weights=[0.1, 0.3, 0.6],  # Fewer Tier 1 schools, more Tier 3
        k=1
    )[0]
    school = random.choice(indian_law_schools[tier])
    return school, tier

def get_school_factors(tier):
    """Get the factors associated with a school tier."""
    return school_tier_factors[tier]

def generate_lawyer_name():
    """Generate a realistic Indian lawyer name using Faker."""
    # Set locale to Indian English for Indian names
    fake = Faker('en_IN')
    return fake.name()

def generate_lawyer_data(num_lawyers=1000):
    """Generate synthetic lawyer data with Indian context."""
    lawyers = []
    
    for lawyer_id in range(1, num_lawyers + 1):
        # Generate dates
        start_practice = random_date(start_date, end_date - timedelta(days=365*2))
        years_exp = (end_date - start_practice).days / 365.25
        
        # Get law school and its tier
        law_school, school_tier = get_random_law_school()
        school_factors = get_school_factors(school_tier)
        
        # Generate awards based on experience and prestige
        awards = get_awards_based_on_experience(years_exp)
        
        # Calculate base skill level with school tier bonus
        base_skill = min(10, max(1, int(np.random.normal(6, 2))))
        skill_with_school = min(10, base_skill + school_factors["skill_bonus"])
        
        # Generate lawyer profile
        lawyer = {
            'lawyer_id': lawyer_id,
            'name': generate_lawyer_name(),
            'law_school': law_school,
            'school_tier': school_tier,
            'graduation_year': start_practice.year - random.randint(0, 5),
            'bar_admission_date': start_practice.strftime('%Y-%m-%d'),
            'start_practice': start_practice.strftime('%Y-%m-%d'),
            'years_experience': round(years_exp, 1),
            'avg_bill_rate': random.randint(5000, 50000) * school_factors["starting_salary_multiplier"],  # Indian Rupees
            'work_rate': random.randint(20, 80),
            'awards': ", ".join(awards),
            'awards_count': len(awards),
            'total_cases': random.randint(50, 500),
            'success_rate': random.uniform(0.5, 0.95) * school_factors["prestige_factor"],
            'total_revenue': random.uniform(1000000, 50000000) * school_factors["starting_salary_multiplier"],  # Indian Rupees
            'avg_revenue_per_case': random.uniform(20000, 200000) * school_factors["starting_salary_multiplier"]  # Indian Rupees
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
            school_factor = school_factors["complex_case_probability"]
            
            # Base frequency adjusted by complexity, experience, and school tier
            base_freq = 5 + (complexity_factor * 10 * exp_factor * school_factor)
            max_freq = remaining - (len(sorted_case_types) - i - 1) * 5
            freq = min(int(base_freq), max_freq)
            
            frequencies.append(freq)
            remaining -= freq
        
        frequencies.append(remaining)
        
        # Add case type percentages
        for (case_type, _), freq in zip(sorted_case_types, frequencies):
            lawyer[f'{case_type.lower().replace(" ", "_")}_percentage'] = freq
        
        lawyers.append(lawyer)
    
    return pd.DataFrame(lawyers)

def generate_case_data(lawyer_df, cases_per_lawyer=5):
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
        
        # Determine lawyer's specialty based on highest percentage case type
        specialty_case_type = max(indian_case_types.keys(), 
                                key=lambda x: lawyer[f'{x.lower().replace(" ", "_")}_percentage'])
        
        # Skill level based on experience, awards, and school tier
        base_skill = min(10, max(1, int(np.random.normal(6, 2))))
        award_bonus = min(2, lawyer['awards_count'] * 0.5)
        school_bonus = school_factors["skill_bonus"]
        skill_level = min(10, base_skill + award_bonus + school_bonus)
        
        for _ in range(num_cases):
            # Higher chance of complex cases for higher tier schools
            if random.random() < school_factors["complex_case_probability"]:
                case_type = random.choice([ct for ct, details in indian_case_types.items() 
                                         if details['complexity'] > 1.2])
            else:
                case_type = random.choice(list(indian_case_types.keys()))
            
            # Get case type details
            case_details = indian_case_types[case_type]
            
            # Case start date
            case_start_date = random_date(start_practice_date, end_date - timedelta(days=30))
            
            # Determine court level based on case type and complexity
            eligible_courts = [court for court, details in indian_court_levels.items() 
                             if case_type in details['case_types']]
            court_weights = [indian_court_levels[level]['complexity_factor'] * 
                           (1 + school_factors["prestige_factor"] - 1) for level in eligible_courts]
            court_level = random.choices(eligible_courts, weights=court_weights, k=1)[0]
            
            # Calculate case duration
            min_duration, max_duration = case_details['duration_range']
            complexity_factor = case_details['complexity'] * indian_court_levels[court_level]['complexity_factor']
            base_duration = random.randint(min_duration, max_duration)
            case_duration = int(base_duration * complexity_factor)
            case_end_date = min(case_start_date + timedelta(days=case_duration), end_date)
            
            # Determine case outcome with school tier and case type influence
            base_success_rate = case_details['success_rate']
            success_probability = (skill_level / 10) * 0.7 + random.random() * 0.3
            if complexity_factor > 1.3:
                success_probability *= 0.9
            success_probability *= school_factors["prestige_factor"]
            success_probability *= base_success_rate
            
            # Select outcome based on probability with three possible outcomes
            if random.random() < success_probability:
                # Higher chance of winning for better lawyers
                if random.random() < 0.7:  # 70% chance of winning if successful
                    outcome = "Won"
                else:
                    outcome = "Settled"
            else:
                outcome = "Lost"
            
            # Calculate billable hours and revenue with school tier influence
            hourly_rate = lawyer['avg_bill_rate']
            billable_hours = case_duration * (0.5 + random.random() * 1.5)
            base_revenue = hourly_rate * billable_hours
            revenue = base_revenue * indian_case_outcomes[outcome]['revenue_multiplier']
            
            case = {
                'case_id': case_id,
                'lawyer_id': lawyer['lawyer_id'],
                'case_type': case_type,
                'court_level': court_level,
                'start_date': case_start_date.strftime('%Y-%m-%d'),
                'end_date': case_end_date.strftime('%Y-%m-%d'),
                'duration_days': (case_end_date - case_start_date).days,
                'result': outcome,
                'billable_hours': round(billable_hours, 1),
                'revenue': round(revenue, 2)
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
    lawyer_df = generate_lawyer_data()
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