import os
import tempfile
from inference import run
import boto3
import uuid
from typing import Optional
import json
import asyncio
from pymongo import MongoClient
import time


os.environ["AWS_DEFAULT_REGION"]='ap-northeast-1'
os.environ["AWS_ACCESS_KEY_ID"]='AKIAWEPYHILPXHGDOTJ5'
os.environ["AWS_SECRET_ACCESS_KEY"]="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"


sqs = boto3.client('sqs')
queue_url = 'https://sqs.ap-northeast-1.amazonaws.com/421964235487/Dublr-Video-Process_Queue'

def list_files_in_bucket(bucket_name):
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket_name)
        files=[]
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append(obj['Key'])
        return files
    except Exception as e:
        print("An error occurred:", e)

async def process_video(vid_key,source_lang,start_time, end_time):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_location = os.path.join(temp_dir,vid_key)
            s3 = boto3.client('s3')
            bucket_name = 'dublr-bucker'
            print('Trying to get object from S3...')
            response = s3.download_file(bucket_name, vid_key, file_location)
            print('Object fetched successfully.')
            if os.path.exists(file_location):
                print("Video File Downloaded and Exists")
            else:
                raise FileNotFoundError("Problem in Uploading Video File")
            exc=run(video_path=file_location, SOURCE_LANG=source_lang, gan=True, lip_sync=True, preset='ultra_fast', start=start_time, end=end_time)
        
        if exc is not None:
            raise exc
        base_name, extension = os.path.splitext(vid_key)
        unique_filename = base_name + "_" + str(uuid.uuid4().hex) + extension
        dubbed_bucket="dublr-dubbed-videos-bucket"
        files=list_files_in_bucket(dubbed_bucket)
        while unique_filename in files:
            unique_filename = base_name + "_" + str(uuid.uuid4().hex) + extension
        s3.upload_file('result/output.mp4',dubbed_bucket, unique_filename)
        print(f"Upload successful: {unique_filename} to {dubbed_bucket}")
        #os.remove('result/output.mp4')
        return unique_filename
    except Exception as e:
        return e

# Continuously poll the queue for new messages
async def main ():

    client = MongoClient("mongodb+srv://umais:61VbcGUEktvxmCB4@cluster0.pyqqhxl.mongodb.net")
    db = client.get_database("user")
    collection = db.get_collection("Video")

    while True:	
        response = sqs.receive_message(	
            QueueUrl=queue_url,		
            MaxNumberOfMessages=1,	
            WaitTimeSeconds=5  # Adjust as needed
        )	
        if 'Messages' in response:	

            video_dubbing_time=None
            try:
                for message in response['Messages']:
                    body = message['Body']
                    receipt_handle = message['ReceiptHandle']
                    data_dict = json.loads(body)
                    jobId = data_dict['jobId']
                    query={'jobId':jobId}
                    result=collection.find_one(query) # Video
                    if result is None:
                        raise FileNotFoundError("No Entry For Video Found")
                    userId=result['userId']
                    originalVideoKey=result['originalVideoKey']
                    sourceLanguage = result['sourceLanguage']
                    startTime = result['startTime']
                    endTime = result['endTime']
                    ###################################################################
                    video_dubbing_time = endTime - startTime
                    ###################################################################
                    update_status={"$set":{"processingStatus":"Processing"}}
                    collection.update_one(query,update_status)
                    req = await process_video(originalVideoKey,sourceLanguage,startTime,endTime)
                    if isinstance(req, str):
                        print("Unique Filename Generated")
                    elif isinstance(req,Exception):
                        raise req
                    current_time_seconds = time.time()
                    current_time_milliseconds = int(current_time_seconds * 1000)
                    update_dubbedVideo={"$set":{"dubbedVideoKey":req,
                                                "processingStatus":"Completed",
                                                "processedDate":current_time_milliseconds}}
                    collection.update_one(query,update_dubbedVideo)
                    sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=receipt_handle)
            except Exception as e:
                if e==FileNotFoundError("No Entry For Video Found"):
                    print("No Such Job or Video Found")
                    sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=receipt_handle)
                else:
                    query={'jobId':jobId}
                    current_time_seconds = time.time()
                    current_time_milliseconds = int(current_time_seconds * 1000)
                    update_failed={"$set":{"processingStatus":"Failed",
                                           "dubbingError":str(e),
                                           "processedDate":current_time_milliseconds}}
                    collection.update_one(query,update_failed)

                    ######################################################################
                    user_query={'userId': userId}
                    time_update={'$inc': {'videoDubbedSeconds': -1*video_dubbing_time}}
                    users_collection = db.get_collection("User")
                    users_collection.update_one(user_query, time_update)
                    ######################################################################

                    sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=receipt_handle)


#client = MongoClient("mongodb+srv://umais:61VbcGUEktvxmCB4@cluster0.pyqqhxl.mongodb.net")
#db = client.get_database("user")
#collection = db.get_collection("Video")
#result=collection.find_one({'jobId': 'fad864ab-479d-467e-9e42-42a830c07cda'})
#print(result)
#client.close()
#cursor = collection.find()
#for document in cursor:
#    print(document)
#client.close()

asyncio.run(main())

