FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image's Yarn apt repo has an expired signing key, which breaks apt-based features.
RUN rm -f /etc/apt/sources.list.d/yarn.list
