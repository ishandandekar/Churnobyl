import boto3

# s3_resource = boto3.resource("s3")
# print(type(s3_resource))
# BUCKET_NAME = "churnobyl"
# churnobyl_bucket = s3_resource.Bucket(BUCKET_NAME)
# print(type(churnobyl_bucket))
# for obj in churnobyl_bucket.objects.all():
#     print(obj.key)

# s3_client = boto3.client("s3")
# BUCKET_NAME = "churnobyl"
# file_name = "hey.log"
# file_path = "./hey.log"
# s3_client.upload_file(file_path, BUCKET_NAME, file_name)

s3_resource = boto3.resource("s3")
# print(type(s3_resource))
BUCKET_NAME = "churnobyl"
churnobyl_bucket = s3_resource.Bucket(BUCKET_NAME)
# print(type(churnobyl_bucket))
for obj in churnobyl_bucket.objects.all():
    print(obj.key)

churnobyl_bucket.upload_file("hey.log", "logs/hey.log")
for obj in churnobyl_bucket.objects.all():
    print(obj.key)
