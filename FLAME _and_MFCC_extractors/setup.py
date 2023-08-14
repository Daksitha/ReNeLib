import setuptools

requirements = open("requirements38.txt").read().splitlines()
# dev_requirements = open("requirements_dev.txt").read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LLB",
    description="Official release ReNeLiB: Real-time Neural Listening Behavior Generation for Socially Interactive Agents- ICMI'23",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daksitha Withanage",
    author_email="daksitha.withanage@gmail.com",
    version="0.0.1",
    packages=["src"],
    package_dir={"": "."},
    install_requires=requirements,
    # extras_require={"dev": dev_requirements},
    python_requires=">=3.8",
)
