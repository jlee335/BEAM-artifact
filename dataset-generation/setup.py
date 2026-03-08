from setuptools import setup, find_packages

setup(
    name="servegen",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "servegen": ["data/*/*.json"],
    },
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
    author="echostone",
    description="ServeGen: realistic LLM workload generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
