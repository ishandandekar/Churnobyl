# Churnobyl

[![Deploy](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml)
[![Tests](https://github.com/ishandandekar/Churnobyl/actions/workflows/tests.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/tests.yaml)

[LICENCE](LICENCE)

> **Warning** : This is a work in progress. Until specified, please do not directly use the code. There will be addtition as well as improvements over the time. Use the code only to get inspiration and not for actual production usage.


## [Contributions](./CONTRIBUTING.md)

Any help is always welcomed. The project is open-source. The key features that are needed to be updated are marked as TODO in readme as well as in code. If you think there can be any other improvement, please make a PR or an issue, and I'll go over it as soon as possible.

### Steps to develop locally:
- Join the org!
- Create a PR
- Get the keys
- Notify owners and reviewers of repo/project
- Push code to branch, some tests must be passes by the branch

## Milestone (18-7-2023): The pipeline ran completely from start to end. No errors while adding artifacts to server too!
## Milestone (7-8-2023): The model serving API WORKED!!!

### Issues:
- custom transformation functions referrenced in [pipeline](./churnobyl/pipeline.py) need to be written again in [api code](./serve/api.py)

### S3 directory structure
```
churnobyl/
├─ api_logs/
│  ├─ predict_logs/
│  ├─ flag_logs/
├─ flagged/
├─ train_logs/
```

### Ideas for monitoring dashboard:

- Prediction rate
- Monthly/daily frequency of requests
- Flag rate
- alert after flags cross certain threshold
- Add probability data

### TODO:
- [ ] Create S3 bucket and folder
- [ ] Run modelling experiments [`notebook`](./notebooks/01-model-experiments.ipynb) one-last-time
- [ ] Add logic for imputing values, or atleast raise error if any
- [ ] Deploy a temporary lambda service and API gateway
- [ ] Create Streamlit dashboard for monitoring
- [ ] Try EvidentlyAI model monitoring service
- [ ] Integrate Streamlit to display graphs

### Refs:

- https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/
- https://www.eliasbrange.dev/posts/deploy-fastapi-on-aws-part-1-lambda-api-gateway/
- https://github.com/eliasbrange/aws-fastapi/tree/main/lambda-api-gateway
- https://blog.searce.com/fastapi-container-app-deployment-using-aws-lambda-and-api-gateway-6721904531d0
- https://www.youtube.com/watch?v=rpVLOVeky6A&ab_channel=VincentStevenson
---
- https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push.html

- https://medium.com/akava/deploying-containerized-aws-lambda-functions-with-terraform-7147b9815599
- https://developer.hashicorp.com/terraform/tutorials/aws/lambda-api-gateway
- https://www.youtube.com/watch?v=VYk3lwZbHBU&ab_channel=TalhaAnwar
- https://www.baeldung.com/ops/dockerfile-env-variable
- https://www.youtube.com/watch?v=9ciujhGdAOs&ab_channel=TechplanetAI
- https://docs.aws.amazon.com/lambda/latest/dg/configuration-envvars.html
- https://www.youtube.com/watch?v=M91vXdjve7A&ab_channel=BeABetterDev
- https://octopus.com/blog/githubactions-docker-ecr
- https://aws.amazon.com/blogs/compute/using-aws-lambda-extensions-to-send-logs-to-custom-destinations/
- https://madewithml.com/courses/mlops/testing/
- https://neptune.ai/blog/ml-model-testing-teams-share-how-they-test-models
- https://github.com/whylabs/whylogs/tree/mainline
- https://whylogs.readthedocs.io/en/stable/examples/integrations/flask_streaming/flask_with_whylogs.html
- https://aws.amazon.com/blogs/compute/deploying-machine-learning-models-with-serverless-templates/
- https://betterprogramming.pub/deploying-ml-models-using-aws-lambda-and-api-gateway-f1f349515c81
- https://aws.amazon.com/blogs/machine-learning/deploy-a-machine-learning-inference-data-capture-solution-on-aws-lambda/
- https://blog.devgenius.io/how-to-put-an-ml-model-into-production-dc3a99eeb2cc
- https://aws.amazon.com/blogs/machine-learning/build-a-ci-cd-pipeline-for-deploying-custom-machine-learning-models-using-aws-services/
- https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://towardsai.net/p/l/model-monitoring-dashboards-made-easy-1-3%3Famp%3D1&ved=2ahUKEwjF06zUrI2AAxVQcGwGHekJDvEQFnoECCcQAQ&usg=AOvVaw1a1kNPwMKpByjAb8PkguQ9
- https://docs.arize.com/arize/quickstart
- https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://neptune.ai/blog/ml-model-monitoring-best-tools&ved=2ahUKEwjF06zUrI2AAxVQcGwGHekJDvEQFnoECCUQAQ&usg=AOvVaw0cWnCVHcV7-lAU9ivcr-lf
- https://retool.com/templates/machine-learning-model-monitoring-dashboard
- https://docs.dominodatalab.com/en/5.3/user_guide/715969/model-monitoring/
- https://stackoverflow.com/questions/65602056/how-to-set-and-access-environment-variables-in-python-file-for-streamlit-app
- https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact