FROM golang:1.24-alpine AS build
RUN apk add --no-cache git
RUN GOBIN=/out go install github.com/minio/minio@RELEASE.2025-04-22T22-12-26Z \
 && GOBIN=/out go install github.com/minio/mc@RELEASE.2025-04-16T18-13-26Z

FROM alpine:3.21
COPY --from=build /out/minio /usr/local/bin/minio
COPY --from=build /out/mc /usr/local/bin/mc
ENTRYPOINT ["/usr/local/bin/minio"]
