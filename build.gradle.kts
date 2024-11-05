import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import io.gitlab.arturbosch.detekt.Detekt
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jlleitschuh.gradle.ktlint.reporter.ReporterType

plugins {
    kotlin("jvm") version "2.0.21"
    kotlin("plugin.serialization") version "2.0.21"
    java
    id("com.github.johnrengelman.shadow") version "8.1.1"
    jacoco
    id("org.jetbrains.dokka") version "1.9.20"
    id("io.gitlab.arturbosch.detekt") version "1.23.7"
    id("com.diffplug.spotless") version "6.25.0"
//    kotlin("jupyter.api") version "0.10.1-8"
    id("com.github.jk1.dependency-license-report") version "2.9"
    id("com.github.spotbugs") version "6.0.26"
    id("org.jlleitschuh.gradle.ktlint") version "12.1.1"
    application
}

group = "jp.live.ugai"
version = "1.0-SNAPSHOT"
// val v = "0.19.0-SNAPSHOT"
val v = "0.30.0"

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
    implementation("org.jetbrains.lets-plot:lets-plot-common:4.5.1")
    implementation("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:4.8.0")
    runtimeOnly("org.slf4j:slf4j-simple:2.0.16")
    implementation("org.apache.commons:commons-math3:3.6.1")
    implementation(kotlin("stdlib"))
    implementation("com.opencsv:opencsv:5.9")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.11.3")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.11.3")
}

tasks {
    compileKotlin {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_1_8)
    }

    compileTestKotlin {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_1_8)
    }

    compileJava {
        options.encoding = "UTF-8"
        sourceCompatibility = "1.8"
        targetCompatibility = "1.8"
    }

    compileTestJava {
        options.encoding = "UTF-8"
        sourceCompatibility = "1.8"
        targetCompatibility = "1.8"
    }

    test {
        useJUnitPlatform()
        finalizedBy(jacocoTestReport) // report is always generated after tests run
    }

    withType<Detekt>().configureEach {
        // Target version of the generated JVM bytecode. It is used for type resolution.
        jvmTarget = "1.8"
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

    withType<ShadowJar> {
        manifest {
            attributes["Main-Class"] = "com.fujitsu.labs.virtualhome.MainKt"
        }
    }
}

ktlint {
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
/*
    additionalEditorconfig.set(
        mapOf(
            "property_naming" to "false"
        )
    )
*/
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
    toolVersion = "0.8.12"
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
        googleJavaFormat("1.24.0") // has its own section below
        formatAnnotations() // fixes formatting of type annotations, see below
    }
}
