import io.gitlab.arturbosch.detekt.Detekt
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jlleitschuh.gradle.ktlint.reporter.ReporterType

plugins {
    kotlin("jvm") version "2.3.0"
    kotlin("plugin.serialization") version "2.3.0"
    java
    id("com.gradleup.shadow") version "9.3.0"
    jacoco
    id("org.jetbrains.dokka") version "2.1.0"
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
    id("com.diffplug.spotless") version "8.2.0"
//    kotlin("jupyter.api") version "0.10.1-8"
    id("com.github.jk1.dependency-license-report") version "3.0.1"
    id("com.github.spotbugs") version "6.4.8"
    id("org.jlleitschuh.gradle.ktlint") version "14.0.1"
    application
}

group = "jp.live.ugai"
version = "1.0-SNAPSHOT"
// val v = "0.19.0-SNAPSHOT"
val v = "0.36.0"

// val ktlint by configurations.creating

repositories {
    mavenCentral()
    maven {
        url = uri("https://oss.sonatype.org/content/repositories/snapshots/")
    }
}

dependencies {
    implementation("ai.djl:basicdataset:$v")
    implementation("ai.djl:api:$v")
//    runtimeOnly("ai.djl.mxnet:mxnet-engine:$v")
    runtimeOnly("ai.djl:model-zoo:$v")
//    runtimeOnly("ai.djl.mxnet:mxnet-model-zoo:$v")

//    runtimeOnly("ai.djl.mxnet:mxnet-native-cu112mkl:1.9.1:linux-x86_64")
    runtimeOnly("ai.djl.mxnet:mxnet-native-mkl:1.9.1:win-x86_64")
    runtimeOnly("ai.djl.mxnet:mxnet-engine:$v")
//    runtimeOnly("ai.djl.pytorch:pytorch-engine:$v")
//    runtimeOnly("ai.djl.pytorch:pytorch-jni:1.12.1-$v")
//    runtimeOnly("ai.djl.pytorch:pytorch-native-cpu:1.12.1")
    //    implementation("ai.djl.pytorch:pytorch-native-cpu:1.12.1:linux-x86_64")
//    runtimeOnly("ai.djl.pytorch:pytorch-native-cu116:1.12.1:linux-x86_64")
    implementation("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:4.12.1")
    implementation("org.jetbrains.lets-plot:lets-plot-common:4.8.2")
    runtimeOnly("org.slf4j:slf4j-simple:2.0.17")
    implementation("org.apache.commons:commons-math3:3.6.1")
    implementation(kotlin("stdlib"))
    implementation("com.opencsv:opencsv:5.12.0")
    testImplementation(platform("org.junit:junit-bom:6.0.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks {
    compileKotlin {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_17)
    }

    compileTestKotlin {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_17)
    }

    compileJava {
        options.encoding = "UTF-8"
        sourceCompatibility = "17"
        targetCompatibility = "17"
    }

    compileTestJava {
        options.encoding = "UTF-8"
        sourceCompatibility = "17"
        targetCompatibility = "17"
    }

    test {
        useJUnitPlatform()
        finalizedBy(jacocoTestReport) // report is always generated after tests run
    }

    withType<Detekt>().configureEach {
        // Target version of the generated JVM bytecode. It is used for type resolution.
        jvmTarget = "17"
        reports {
            // observe findings in your browser with structure and code snippets
            html.required.set(true)
            // checkstyle like format mainly for integrations like Jenkins
            xml.required.set(true)
            // similar to the console output, contains issue signature to manually edit baseline files
            txt.required.set(true)
            // standardized SARIF format (https://sarifweb.azurewebsites.net/) to support integrations
            // with Github Code Scanning
            sarif.required.set(true)
        }
    }

    check {
        dependsOn("ktlintCheck")
    }

    jacocoTestReport {
        reports {
            xml.required = true
            html.required = false
        }
        dependsOn(test) // tests are required to run before generating the report
    }

    shadowJar {
        manifest {
            attributes["Main-Class"] = "com.fujitsu.labs.virtualhome.MainKt"
        }
    }

    register<JavaExec>("execute") {
        classpath = sourceSets["main"].runtimeClasspath
        mainClass.set(
            if (project.hasProperty("mainClass")) {
                project.property("mainClass") as String
            } else {
                "com.fujitsu.labs.virtualhome.MainKt"
            },
        )
    }
}

ktlint {
    version = "1.6.0"
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
    }
}

detekt {
    buildUponDefaultConfig = true // preconfigure defaults
    allRules = false // activate all available (even unstable) rules.
    // point to your custom config defining rules to run, overwriting default behavior
    config.from(files("$projectDir/config/detekt.yml"))
}

spotbugs {
    ignoreFailures.set(true)
}

jacoco {
    toolVersion = "0.8.13"
//    reportsDirectory.set(layout.buildDirectory.dir("customJacocoReportDir"))
}

application {
    mainClass.set("jp.live.ugai.d2j.BatchNorm2Kt")
}

spotless {
    java {
        target("src/*/java/**/*.java")
        // Use the default importOrder configuration
        importOrder()
        removeUnusedImports()

        // Choose one of these formatters.
        googleJavaFormat("1.28.0") // has its own section below
        formatAnnotations() // fixes formatting of type annotations, see below
    }
}

dokka.dokkaSourceSets {
    configureEach {
        jdkVersion.set(17)
        enableJdkDocumentationLink.set(false)
        enableKotlinStdLibDocumentationLink.set(false)
    }
}
