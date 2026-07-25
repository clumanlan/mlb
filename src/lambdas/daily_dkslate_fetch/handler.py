from daily_dkslate_fetch import fetch_and_save_daily_slates
import json
from datetime import datetime

def lambda_handler(event, context):
    date = event.get('date') or datetime.now().strftime('%Y-%m-%d')
    try:
        result = fetch_and_save_daily_slates(date)
        
        # Make sure we return something meaningful
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed MLB slate data',
                'date': date,
                'result': result['draft_groups_processed']
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


