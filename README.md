# Churnobyl

[![Deploy](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml)

[![Configure AWS](https://github.com/ishandandekar/Churnobyl/actions/workflows/aws_configure.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/aws_configure.yaml)

[![Tests](https://github.com/ishandandekar/Churnobyl/actions/workflows/tests.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/tests.yaml)

[LICENCE](LICENCE)

> **Warning**: This is a work in progress. Until specified, please do not directly use the code. There will be addtition as well as improvements over the time. Use the code only to get inspiration and not for actual production usage.

### Milestone (18-7-2023): The pipeline ran completely from start to end. No errors while add artifacts to server too!

## Contributions

Any help is always welcomed. The project is open-source. The key features that are needed to be updated are marked as TODO in readme as well as in code. If you think there can be any other improvement, please make a PR or an issue, and I'll go over it as soon as possible.

### TODO:

- [x] Setup project
- [x] Integrate python environment
- [x] Learn pre-commit and linters
- [x] Make base notebooks without thinking of optimization
- [x] Create a `pipeline.py` script
- [x] Explore and update the files in `./data_schema`
- [x] Only keep `.yaml` file for schema
- [x] Figure out preprocessing code present in [`./temp/temp.py`](./temp/temp.py)
- [x] Run model experiments notebook
- [x] Add code to save preprocessor and all models into the specified directory
- [x] Fill the [`./churnobyl/data.py`](./churnobyl/data.py)
- [x] Fill the [`./churnobyl/visualize.py`](./churnobyl/visualize.py)
- [x] Fill the [`./churnobyl/model.py`](./churnobyl/model.py)
- [x] Update config file
- [x] Add `great_expectations` instead of `pandera`
- [x] Sign in to prefect workspace
- [x] Fill the [`./churnobyl/pipeline.py`](./churnobyl/pipeline.py), to reflect the actual workflow
- [ ] Add tests for `pytest`
- [x] Modulize the notebook code into python scripts
- [ ] Run modelling experiments [`notebook`](./notebooks/01-model-experiments.ipynb) one-last-time
- [x] Integrate prefect for workflow orchestration
- [ ] Create `Dockerfile`
- [ ] Add workflow files using Github Actions
- [ ] Explore AWS services to think for deployment options
- [ ] Try EvidentlyAI model monitoring service
- [ ] Integrate Streamlit to display graphs

### Refs:

- https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/
- https://www.eliasbrange.dev/posts/deploy-fastapi-on-aws-part-1-lambda-api-gateway/
- https://github.com/eliasbrange/aws-fastapi/tree/main/lambda-api-gateway

---

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
