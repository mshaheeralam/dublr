import os
import tempfile
from inference import run
import boto3
import uuid
import json
import asyncio
from pymongo import MongoClient
import time
from dotenv import load_dotenv
from emails import send_email
import gc
import torch

load_dotenv()
sqs = boto3.client('sqs')
queue_url = os.getenv("QUEUE_URL")


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


async def process_video(vid_key,lip_flag,sub_flag,source_lang,start_time, end_time):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_location = os.path.join(temp_dir,vid_key)
            s3 = boto3.client('s3')
            bucket_name = 'dublr-bucket'
            print('Trying to get object from S3...')
            response = s3.download_file(bucket_name, vid_key, file_location)
            print('Object fetched successfully.')
            if os.path.exists(file_location):
                print("Video File Downloaded and Exists")
            else:
                raise FileNotFoundError("Problem in Uploading Video File")
            #print(source_lang)
            gc.collect()
            torch.cuda.empty_cache()
            start = time.time()
            if source_lang=="auto":
                exc=run(video_path=file_location, lip_sync=lip_flag, subtitiles=sub_flag, start=start_time, end=end_time)
            else:
                exc=run(video_path=file_location, lip_sync=lip_flag, subtitiles=sub_flag, start=start_time, end=end_time, SOURCE_LANG=source_lang)
            end = time.time()
            print("Total :", (end-start) / 60, "min\n")

        if exc is not None:
            raise exc
        base_name, _ = os.path.splitext(vid_key)
        unique_filename = vid_key
        dubbed_bucket="dubbed-bucket"
        files=list_files_in_bucket(dubbed_bucket)
        while unique_filename in files:
            unique_filename = base_name + "_" + str(uuid.uuid4().hex)
        s3.upload_file('output.mp4',dubbed_bucket, unique_filename)
        print(f"Upload successful: {unique_filename} to {dubbed_bucket}")
        os.remove('output.mp4')
        return unique_filename
    except Exception as e:
        return e



# Continuously poll the queue for new messages
async def main ():
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("user")
    collection = db.get_collection("Video")

    while True:
        #print("in main")	
        response = sqs.receive_message(	
            QueueUrl=queue_url,		
            MaxNumberOfMessages=1,	
            WaitTimeSeconds=5  # Adjust as needed
        )
       
        if 'Messages' in response:	

            video_dubbing_time=None
            status=None
            try:
                for message in response['Messages']:
                    body = message['Body']
                    receipt_handle = message['ReceiptHandle']
                    data_dict = json.loads(body)
                    jobId = data_dict['jobId']
                    query={'jobId':jobId}
                    result=collection.find_one(query) # Video
                    #print("result", result)
                    if result is None:
                        raise FileNotFoundError("No Entry For Video Found")
                    userId=result['userId']
                    video_name=result['name']
                    originalVideoKey=result['originalVideoKey']
                    sourceLanguage = result['sourceLanguage']
                    lip_flag=result['lipSync']
                    sub_flag=result['generateSubtitles']
                    startTime = result['startTime']
                    endTime = result['endTime']
                    ###################################################################
                    video_dubbing_time = endTime - startTime
                    ###################################################################
                    update_status={"$set":{"processingStatus":"Processing"}}
                    user_query={'_id': userId}
                    users_collection = db.get_collection("User")
                    result=users_collection.find_one(user_query)
                    email=result['email']
                    print(f"\nEmail: {email}\n")
                    collection.update_one(query,update_status)
                    req = await process_video(originalVideoKey,lip_flag,sub_flag,sourceLanguage,startTime,endTime)
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
                    # TEMP EMAIL SEND
                    # sqs.send_message(	
                    #     QueueUrl=os.getenv("EMAIL_QUEUE"),		
                    #     MessageBody=json.dumps({"status": "success","video_name": video_name ,"receiver":email}),	
                    # )      
                    send_email(status="success", video_name=video_name, RECIPIENT=email)
            except Exception as e:
                if str(e)=="No Entry For Video Found":
                    print("No Such Job or Video Found")
                    sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=receipt_handle)
                else:
                    status="error"
                    if str(e)=="Multiple speaker detected":
                        status='multi_speaker_error'
                        print('Multiple speaker detected')
                    elif str(e)=="Could Not Create Segments":
                        status='segmentation_error'
                        print("Could Not Create Segments")
                    query={'jobId':jobId}
                    current_time_seconds = time.time()
                    current_time_milliseconds = int(current_time_seconds * 1000)
                    update_failed={"$set":{"processingStatus":"Failed",
                                           "dubbingError":str(e),
                                           "processedDate":current_time_milliseconds}}
                    collection.update_one(query,update_failed)

                    ######################################################################
                    user_query={'_id': userId}
                    time_update={'$inc': {'videoDubbedSeconds': -1*video_dubbing_time}}
                    users_collection = db.get_collection("User")
                    users_collection.update_one(user_query, time_update)
                    ######################################################################

                    sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=receipt_handle)

                    # sqs.send_message(	
                    # QueueUrl=os.getenv("EMAIL_QUEUE"),		
                    # MessageBody=json.dumps({"status": status,"video_name": video_name, "receiver":email}),	
                    # ) 
                    send_email(status, video_name=video_name ,RECIPIENT=email)

#user_query={'email': 'afzalmengal54@gmail.com'}
#time_update={'$inc': {'videoDubbedSeconds': -1*58}}
#client = MongoClient(os.getenv("MONGO_URL"))
#db = client.get_database("user")
#collection = db.get_collection("User")
#collection.update_one(user_query,time_update)
#cursor = collection.find()
#for document in cursor:
#   print(document)
#client.close()

#send_email(RECIPIENT="afzalmengal54@gmail.com",text="""<p>Your video has been dubbed successfully and is now available in your library.</p>
#                        <p>You can access it right away by logging into your account on our website.</p>
#                        <p>Your feedback is valuable to us, and we would greatly appreciate it if you could take a few moments to share your thoughts on the feedback form present on our website. Your input helps us enhance our services and provide you with an even better experience.</p>""")
asyncio.run(main())
#send_email(status="success", RECIPIENT="aliagha135@hotmail.com")
#send_email(RECIPIENT="afzalmengal54@gmail.com",message="Hello")
